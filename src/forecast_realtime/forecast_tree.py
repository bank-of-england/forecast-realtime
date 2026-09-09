"""Nested forecast trees.

A ``TreeNode`` describes one node of a tree used to produce forecasts. Leaves
are bare ``ForecastModel`` instances (identified by their ``label``); internal
nodes are nested ``TreeNode`` instances (identified by their ``name``). Each
node stores a ``transform`` that produces its output from its direct
``children``. ``ForecastTree`` fits the whole tree and returns the root's
output.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import pandas as pd

from forecast_realtime._model_data import ModelData
from forecast_realtime.data_transformation import (
    DataTransformationPipeline,
    FittedDataTransformation,
)
from forecast_realtime.forecast_model import (
    FittedModelConfiguration,
    ForecastContext,
    ForecastModel,
    ForecastResult,
)

TransformType = Callable[[dict[str, pd.DataFrame]], pd.DataFrame] | ForecastModel


# --------------------------------------------------------------------------- #
# Component helpers, shared by the node evaluators below.                      #
# --------------------------------------------------------------------------- #
def _as_frame(values: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Coerce fitted/forecast values to a DataFrame, keeping their column name(s)."""
    return values.to_frame() if isinstance(values, pd.Series) else values


def _leaf_in_sample_frame(leaf: ForecastModel) -> pd.DataFrame:
    """Return a leaf's in-sample output, falling back to fitted target history."""
    try:
        return _as_frame(leaf.fitted_values)
    except AttributeError:
        return _as_frame(leaf.y)


def _node_transform_kwargs(
    kwargs: dict, transform: ForecastModel, *, fitted: bool = False
) -> dict:
    """Return keyword arguments for a stacking model transform."""
    mapping = (
        transform._fitted_model_configuration.data_transformation.data_transformation
        if fitted
        else transform.data_transformation
    )
    blocked = {"y_lags", "X_lags", "dummies", "X_imputation"}
    if mapping is None:
        blocked.update(("frequency", "drop_transformation_nans"))
    return {k: v for k, v in kwargs.items() if k not in blocked} | {
        "data_transformation": None
    }


def _labelled_components(node, components, target_data):
    """Label each actual child output without guessing callable arithmetic units."""
    for name, child in zip(node.child_names, node.children, strict=True):
        frame = components[name]
        model = child if isinstance(child, ForecastModel) else child.transform
        if isinstance(model, ForecastModel):
            metrics = model.native_metric_mapping(list(frame.columns))
            frequency = model._fitted_model_configuration.data_transformation.frequency
            raw_frequencies = model._raw_data.frequencies("y")
            frequencies = {
                column: raw_frequencies.get(column) or frequency
                for column in frame.columns
            }
        else:
            metrics = {column: "levels" for column in frame.columns}
            frequencies = target_data.frequencies("y")
        yield name, frame, metrics, frequencies


def _select_target(frame: pd.DataFrame, target: str, source_name: str) -> pd.DataFrame:
    """Reduce a source frame to the single ``target`` column.

    Selects ``target`` when present, otherwise accepts a single-column frame
    and renames it to ``target`` so every value handed to a callable transform
    is a single-column DataFrame named after the target (allowing arithmetic to
    align across children).
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"source {source_name!r} produced a {type(frame).__name__}; "
            "expected a pandas DataFrame"
        )
    if target in frame.columns:
        return frame[[target]]
    if frame.shape[1] == 1:
        return frame.rename(columns={frame.columns[0]: target})
    raise ValueError(
        f"source {source_name!r} produced a forecast with columns "
        f"{list(frame.columns)}; expected the target column {target!r} "
        "or a single column"
    )


def _resolve_node_target(node_target: str | None, y: pd.DataFrame) -> str:
    """Resolve the target column a callable node reduces its children to.

    Uses ``node_target`` when given (must be a column of ``y``), else the sole
    column of ``y`` when unambiguous.
    """
    if node_target is not None:
        if node_target not in y.columns:
            raise ValueError(
                f"target {node_target!r} is not a column of y; "
                f"got columns {list(y.columns)}"
            )
        return node_target
    if y.shape[1] == 1:
        return y.columns[0]
    raise ValueError(
        "a callable transform needs a target: y has more than one column and no "
        "target was set on the node; pass target=<column name> to TreeNode(...)."
    )


def _reduce_components(
    components: dict[str, pd.DataFrame], target: str
) -> dict[str, pd.DataFrame]:
    """Reduce every child frame to the node's single ``target`` column."""
    return {
        name: _select_target(frame, target, name) for name, frame in components.items()
    }


@dataclass
class TreeNode:
    """A single node in a nested forecast tree.

    Parameters
    ----------
    transform : Callable or ForecastModel
        Either a function ``dict[str, pd.DataFrame] -> pd.DataFrame`` that
        produces this node's output from its children (each reduced to the
        node's ``target`` column), or a ``ForecastModel`` that stacks: it is
        fitted on its children's raw components (one column per leaf/node name)
        and its own ``forecast()`` produces this node's output. Validated and
        stored here; invoked/fitted by ``ForecastTree``.
    children : list of ForecastModel or TreeNode
        The direct children of this node. Must be a non-empty ``list``.
    name : str, optional
        Identifier for this node. Defaults to ``"node"`` when ``None``.
    target : str, optional
        The ``y`` column a **callable** ``transform`` reduces its children to
        (and the tree forecasts, when this is the root). Auto-resolved when
        ``y`` has one column. Must not be set when ``transform`` is a
        ``ForecastModel`` (which picks its own target via its formula).

    Notes
    -----
    The same object may be referenced through multiple branches (DAG-style
    reuse); ``all_leaves()`` and ``nodes()`` deduplicate by object identity.
    Rejected: two *distinct* objects sharing a name anywhere in the tree, or
    duplicate names among the direct children of a single node.

    Raises
    ------
    TypeError
        If ``transform`` is neither callable nor a ``ForecastModel``, or any
        child is neither a ``ForecastModel`` nor a ``TreeNode``.
    ValueError
        If ``children`` is not a non-empty ``list``; if ``target`` is set on a
        ``ForecastModel`` transform; if direct children of any node do not have
        unique names; if two distinct objects share a name; or if the tree
        contains a cycle.
    """

    transform: TransformType
    children: list[ForecastModel | TreeNode]
    name: str | None = None
    target: str | None = None

    def __post_init__(self) -> None:
        if not (callable(self.transform) or isinstance(self.transform, ForecastModel)):
            raise TypeError(
                "transform must be callable or a ForecastModel instance; got "
                f"{type(self.transform).__name__}"
            )

        if self.target is not None and isinstance(self.transform, ForecastModel):
            raise ValueError(
                "target must not be set when transform is a ForecastModel; a "
                "ForecastModel transform picks its own target via its formula."
            )

        if not isinstance(self.children, list):
            raise ValueError(
                f"children must be a list of ForecastModel/TreeNode instances; "
                f"got {type(self.children).__name__}"
            )
        if len(self.children) == 0:
            raise ValueError(
                "children must be a non-empty list of ForecastModel/TreeNode instances"
            )

        for child in self.children:
            if not isinstance(child, (ForecastModel, TreeNode)):
                raise TypeError(
                    "each element of children must be a ForecastModel or TreeNode "
                    f"instance; got {type(child).__name__}"
                )

        if self.name is None:
            self.name = "node"

        name_to_ids: dict[str, set[int]] = {}
        self._traverse([], [], name_to_ids, set(), set(), frozenset())
        self._raise_on_name_collisions(name_to_ids)

    @property
    def child_names(self) -> list[str]:
        """Names of the direct children, in order (leaf ``label`` / node ``name``)."""
        return [
            child.label if isinstance(child, ForecastModel) else child.name
            for child in self.children
        ]

    def all_leaves(self) -> list[ForecastModel]:
        """All unique leaf ``ForecastModel`` instances, in first-occurrence order."""
        all_leaves_result: list[ForecastModel] = []
        name_to_ids: dict[str, set[int]] = {}
        self._traverse(all_leaves_result, [], name_to_ids, set(), set(), frozenset())
        self._raise_on_name_collisions(name_to_ids)
        return all_leaves_result

    def nodes(self) -> list[TreeNode]:
        """All ``TreeNode`` nodes in dependency order (children before parents)."""
        nodes_result: list[TreeNode] = []
        name_to_ids: dict[str, set[int]] = {}
        self._traverse([], nodes_result, name_to_ids, set(), set(), frozenset())
        self._raise_on_name_collisions(name_to_ids)
        return nodes_result

    def _traverse(
        self,
        all_leaves_result: list[ForecastModel],
        nodes_result: list[TreeNode],
        name_to_ids: dict[str, set[int]],
        visited_node_ids: set[int],
        visited_leaf_ids: set[int],
        path_ids: frozenset,
    ) -> None:
        """Traverse the tree in post-order and collect names and objects."""
        node_id = id(self)
        if node_id in path_ids:
            raise ValueError(f"Cycle detected in TreeNode tree at node {self.name!r}")
        if node_id in visited_node_ids:
            return
        path_ids = path_ids | {node_id}

        sibling_names = self.child_names
        if len(set(sibling_names)) != len(sibling_names):
            raise ValueError(
                f"Duplicate child names among direct children of node {self.name!r}: "
                f"{sibling_names}"
            )

        name_to_ids.setdefault(self.name, set()).add(node_id)

        for child in self.children:
            if isinstance(child, ForecastModel):
                leaf_id = id(child)
                name_to_ids.setdefault(child.label, set()).add(leaf_id)
                if leaf_id not in visited_leaf_ids:
                    visited_leaf_ids.add(leaf_id)
                    all_leaves_result.append(child)
            else:
                child._traverse(
                    all_leaves_result,
                    nodes_result,
                    name_to_ids,
                    visited_node_ids,
                    visited_leaf_ids,
                    path_ids,
                )

        visited_node_ids.add(node_id)
        nodes_result.append(self)

    @staticmethod
    def _raise_on_name_collisions(name_to_ids: dict[str, set[int]]) -> None:
        """Raise if any name maps to more than one distinct object id."""
        collisions = sorted(name for name, ids in name_to_ids.items() if len(ids) > 1)
        if collisions:
            raise ValueError(f"Duplicate names found in TreeNode tree: {collisions}")


class ForecastTree(ForecastModel):
    """Forecast model that produces the root output of a ``TreeNode`` tree.

    Every leaf is fitted on the shared raw ``y``/``X`` (``**kwargs`` forwarded
    to each leaf's ``fit()``), and each node's output is produced bottom-up by
    its ``transform``. Intermediate forecasts are kept on the instance for
    inspection: ``leaf_forecasts_`` (by leaf ``label``) and ``node_forecasts_``
    (by node ``name``).

    Parameters
    ----------
    spec : TreeNode
        The (possibly nested) tree describing which models to fit and how each
        node's output is produced. A callable node's ``target`` (see
        ``TreeNode``) selects the column it reduces its children to.
    label : str | None
        Label for this model instance; passed through to ``ForecastModel``.
    data_transformation : dict[str, str] | None
        Optional tree-owned transformation configuration, accepting the same
        values as ``ForecastModel``. Used as the fallback ``data_transformation``
        for every leaf/nested tree that has no model-owned ``data_transformation``
        of its own; a leaf's/nested tree's own pipeline still takes precedence,
        and a nested tree resolves this same fallback rule recursively for its
        own leaves/children. The call-level ``data_transformation`` passed to
        ``fit()``/``forecast()`` is used only where neither this tree nor any
        nearer ancestor tree owns a pipeline.
    """

    def __init__(
        self,
        spec: TreeNode,
        label: str | None = None,
        data_transformation: dict[str, str] | None = None,
    ):
        if not isinstance(spec, TreeNode):
            raise TypeError(
                f"spec must be a TreeNode instance; got {type(spec).__name__}"
            )
        self.spec = spec
        super().__init__(
            label=label,
            data_transformation=data_transformation,
        )
        self._refresh_capability_flags()

    def forecast(
        self,
        steps: int = 1,
        X: pd.DataFrame | None = None,
        y: pd.DataFrame | None = None,
        decomp: bool = False,
        context: ForecastContext | None = None,
        **kwargs,
    ) -> ForecastResult:
        """Retain the tree's positional context on the shared forecast path."""
        return super().forecast(
            steps=steps, X=X, y=y, decomp=decomp, context=context, **kwargs
        )

    def native_metric_mapping(
        self, target_variables: list[str] | None = None
    ) -> dict[str, str]:
        """Return the metric space used by the fitted root output."""
        target_variables = list(target_variables or self.y.columns)
        root_transform = self.spec.transform
        if isinstance(root_transform, ForecastModel):
            return root_transform.native_metric_mapping(
                target_variables=list(root_transform.y.columns)
            )
        return {variable: "levels" for variable in target_variables}

    def resolve_target_variables(self, y_variables: list[str]) -> list[str]:
        """Return the targets selected by the root transform."""
        root_transform = self.spec.transform
        if isinstance(root_transform, ForecastModel):
            return root_transform.resolve_target_variables(y_variables)
        if self.spec.target is not None:
            return [self.spec.target]
        return list(y_variables)

    def resolve_input_data_transformation(
        self,
        data_transformation: dict[str, str] | None = None,
        *,
        y_variables: list[str] | None = None,
        X_variables: list[str] | None = None,
    ) -> DataTransformationPipeline | None:
        """Resolve a fallback pipeline without validating the shared panel."""
        mapping = self.data_transformation
        if mapping is None:
            mapping = data_transformation
        return DataTransformationPipeline(mapping) if mapping is not None else None

    def _refresh_capability_flags(self) -> None:
        """Derive aggregate preprocessing capabilities from tree components."""
        components = list(self.spec.all_leaves())
        components.extend(
            node.transform
            for node in self.spec.nodes()
            if isinstance(node.transform, ForecastModel)
        )
        self._needs_ragged_edge_imputation = any(
            component._needs_ragged_edge_imputation for component in components
        )
        self._handles_missing_values = all(
            component._handles_missing_values for component in components
        )

    def input_requirements(
        self,
        y_variables: list[str],
        X_variables: list[str] | None = None,
        data_transformation=None,
    ):
        """Compose consumer requests without merging roles or losing unmapped inputs."""
        fallback = (
            self.data_transformation
            if self.data_transformation is not None
            else data_transformation
        )
        requirements = []
        for leaf in self.spec.all_leaves():
            requirements.extend(
                leaf.input_requirements(y_variables, X_variables, fallback)
            )

        def output_columns(component):
            if isinstance(component, ForecastModel):
                return component.resolve_target_variables(y_variables)
            if isinstance(component.transform, ForecastModel):
                return component.transform.resolve_target_variables(y_variables)
            return [component.target or y_variables[0]]

        for node in self.spec.nodes():
            if not isinstance(node.transform, ForecastModel):
                continue
            columns = []
            for name, child in zip(node.child_names, node.children, strict=True):
                outputs = output_columns(child)
                columns.extend(
                    [name]
                    if len(outputs) == 1
                    else [f"{name}_{column}" for column in outputs]
                )
            for request in node.transform.input_requirements(y_variables, columns):
                requirements.append(
                    replace(
                        request,
                        consumer=f"{self.label}/{node.name}/{request.consumer}",
                        X_kind="component",
                        X=request.X
                        if request.explicit
                        else tuple((column, "native") for column, _ in request.X),
                    )
                )
        return tuple(requirements)

    def input_metric_requirements(
        self, y_variables, X_variables=None, data_transformation=None
    ):
        """Project consumer requests for callers of the established inspection API."""
        metrics = {}
        for request in self.input_requirements(
            y_variables, X_variables, data_transformation
        ):
            for role in ("y", "X"):
                for variable, metric in request.items(role):
                    metrics.setdefault(variable, set()).add(metric)
        return {variable: tuple(sorted(values)) for variable, values in metrics.items()}

    required_input_metrics = input_metric_requirements

    def _resolve_child_data_transformation(
        self, kwargs: dict, *, fitted: bool = False
    ) -> dict:
        """Override the ``data_transformation`` fallback forwarded to children.

        When this tree owns a ``data_transformation``, its mapping
        becomes the ``data_transformation`` fallback forwarded to every leaf
        and nested tree that has no pipeline of its own (a leaf's/nested
        tree's own pipeline still wins); otherwise the call-level
        ``data_transformation`` already in ``kwargs`` is left untouched.
        """
        if fitted:
            policy = self._fitted_model_configuration.data_transformation
            mapping = (
                dict(policy.data_transformation)
                if policy.pipeline_source == "model"
                else None
            )
        else:
            mapping = self.data_transformation
        if mapping is None:
            return kwargs
        return {**kwargs, "data_transformation": mapping}

    def _resolve_fit_origin(self, root_transform: TransformType):
        """Resolve the final usable date represented by the fitted root output."""
        if isinstance(root_transform, ForecastModel):
            return root_transform.last_y_fit_date

        root_output = _as_frame(self.fitted_values_).dropna(how="any")
        if root_output.empty:
            raise ValueError(
                "The callable root transform produced no usable fitted output."
            )
        return root_output.index[-1]

    def _commit_fit(self, candidate: ForecastTree) -> None:
        """Publish fitted state from ``candidate`` to this tree and its components."""
        original_spec = self.spec
        candidate_state = {
            key: value for key, value in candidate.__dict__.items() if key != "spec"
        }
        self.__dict__.clear()
        self.__dict__.update(candidate_state)
        self.spec = original_spec

        visited: set[tuple[int, int]] = set()
        self._commit_component_states(original_spec, candidate.spec, visited)

    @classmethod
    def _commit_component_states(
        cls,
        original_node: TreeNode,
        candidate_node: TreeNode,
        visited: set[tuple[int, int]],
    ) -> None:
        """Publish fitted state for corresponding models in a tree graph."""
        pair = (id(original_node), id(candidate_node))
        if pair in visited:
            return
        visited.add(pair)

        original_transform = original_node.transform
        candidate_transform = candidate_node.transform
        if isinstance(original_transform, ForecastModel):
            transform_pair = (id(original_transform), id(candidate_transform))
            if transform_pair not in visited:
                visited.add(transform_pair)
                original_transform._commit_fit(candidate_transform)

        for original_child, candidate_child in zip(
            original_node.children, candidate_node.children
        ):
            if isinstance(original_child, TreeNode):
                cls._commit_component_states(original_child, candidate_child, visited)
            else:
                child_pair = (id(original_child), id(candidate_child))
                if child_pair not in visited:
                    visited.add(child_pair)
                    original_child._commit_fit(candidate_child)

    def _fit_data_impl(self, data: ModelData, **kwargs):
        """Fit every leaf and every stacking transform in the spec tree.

        ``y``/``X`` are forwarded to every leaf/nested tree raw
        (untransformed); each leaf/nested tree resolves and applies its own
        transformation. ``**kwargs`` (e.g. ``y_lags``/``X_lags``/``dummies``,
        ``data_transformation``/``frequency``/``X_imputation``/
        ``drop_transformation_nans``) are forwarded to each
        leaf's ``fit()``, with ``data_transformation`` overridden by this
        tree's own pipeline when it has one (see
        ``_resolve_child_data_transformation``).
        """
        self._raw_data = data.history()
        y = data.to_wide("y")

        self._refresh_capability_flags()
        kwargs = self._resolve_child_data_transformation(kwargs)
        self._fit(y, data.to_wide("X"), **kwargs)

        root_transform = self.spec.transform
        if isinstance(root_transform, ForecastModel):
            # The root's ForecastModel transform already picked its target(s)
            # via its own formula/y; defer to it.
            self.y = root_transform.y
        else:
            self.y = y[[self._node_targets[self.spec.name]]]
        self.y_name = self.y.columns[0]
        self._n_output_cols = self.y.shape[1]
        self.y_lags = 0
        self.X_lags = 0
        # Mirror the base ForecastModel.fit contract: the last training date is
        # read by RealTimeModel (e.g. to anchor X imputation over the horizon).
        self.last_y_fit_date = self._resolve_fit_origin(root_transform)
        self._fitted_model_configuration = FittedModelConfiguration(
            data_transformation=FittedDataTransformation.from_fit(
                self.resolve_input_data_transformation(kwargs.get("data_transformation")),
                y_variables=list(self.y.columns),
                X_variables=None,
                frequency=None,
                X_imputation=None,
                pipeline_source=(
                    "model" if self.data_transformation is not None else "fallback"
                ),
            ),
            y_columns=tuple(self.y.columns),
            X_columns=None,
            y_lags=0,
            X_lags=0,
            dummies=None,
            dummy_definitions=None,
            dummy_columns=(),
            forecast_origin=(
                self.last_y_fit_date.to_timestamp(how="end").normalize()
                if isinstance(self.last_y_fit_date, pd.Period)
                else pd.Timestamp(self.last_y_fit_date)
            ),
            drop_transformation_nans=True,
        )
        self._is_fitted = True
        return self

    def _predict_data(
        self,
        data: ModelData,
        *,
        forecast_origin=None,
        steps=1,
        decomp=False,
        **kwargs,
    ) -> ForecastResult:
        """Evaluate each consumer from shared raw observations and fitted policies."""
        if not isinstance(steps, int) or steps <= 0:
            raise ValueError("'Steps' must be an integer greater than zero")

        kwargs = self._resolve_child_data_transformation(kwargs, fitted=True)
        forecast_origin = (
            forecast_origin if forecast_origin is not None else data.index("y")[-1]
        )
        if type(self)._forecast is ForecastTree._forecast:
            forecast = self._forecast_data(
                data, steps=steps, forecast_origin=forecast_origin, **kwargs
            )
            conditioning = {
                "X": data.to_wide("X", "conditioning"),
                "y": data.to_wide("y", "conditioning"),
            }
        else:
            context = ForecastContext._from_data(data, forecast_origin)
            conditioning = {"X": context.X_conditioning, "y": context.y_conditioning}
            forecast = self._forecast(
                context=context,
                steps=steps,
                **conditioning,
                forecast_origin=forecast_origin,
                **kwargs,
            )
        return self._finalise_forecast(
            forecast,
            steps=steps,
            forecast_origin=forecast_origin,
            decomp=decomp,
            decomp_kwargs={
                **conditioning,
                "forecast_origin": forecast_origin,
                **kwargs,
            },
        )

    def _fit(self, y: pd.DataFrame, X: pd.DataFrame | None = None, **kwargs):
        for leaf in self.spec.all_leaves():
            leaf_data = self._raw_data
            leaf_formula = getattr(leaf, "_formula", None)
            if leaf_formula:
                target_rows = leaf_formula.extract_y(y).notna().all(axis=1)
                leaf_data = leaf_data.with_dates("y", y.index[target_rows])
            leaf._fit_from_data(leaf_data, **kwargs)

        nodes = self.spec.nodes()
        # Resolve the column each callable node reduces its children to.
        self._node_targets: dict[str, str] = {
            node.name: _resolve_node_target(node.target, y)
            for node in nodes
            if not isinstance(node.transform, ForecastModel)
        }

        # Evaluate each node bottom-up on the leaves' in-sample fitted values,
        # fitting each stacking transform on its children's components and
        # materialising callable-node in-sample outputs along the same path as
        # forecast-time evaluation.
        raw: dict[str, pd.DataFrame] = {
            leaf.label: _leaf_in_sample_frame(leaf) for leaf in self.spec.all_leaves()
        }
        for node in nodes:
            components = {name: raw[name] for name in node.child_names}
            transform = node.transform
            if isinstance(transform, ForecastModel):
                node_data = ModelData.from_components(
                    self._raw_data,
                    _labelled_components(node, components, self._raw_data),
                )
                X_node = node_data.to_wide("X").dropna(how="all")
                common_index = X_node.index.intersection(y.index)
                node_data = node_data.with_dates("y", common_index).with_dates(
                    "X", common_index
                )
                try:
                    transform._fit_from_data(
                        node_data,
                        **_node_transform_kwargs(kwargs, transform),
                    )
                except ValueError as error:
                    raise ValueError(f"node {node.name!r}: {error}") from error
                raw[node.name] = _as_frame(transform.fitted_values)
            else:
                reduced = _reduce_components(components, self._node_targets[node.name])
                raw[node.name] = transform(reduced)

        self.fitted_values_ = raw[self.spec.name]
        return self

    def _forecast(
        self,
        context: ForecastContext,
        steps: int = 1,
        X: pd.DataFrame | None = None,
        y: pd.DataFrame | None = None,
        forecast_origin=None,
        **kwargs,
    ) -> pd.DataFrame:
        """Adapt the established context hook to the shared tree traversal."""
        return self._forecast_data(
            ModelData.from_context(context, self._raw_data),
            steps=steps,
            forecast_origin=forecast_origin,
            **kwargs,
        )

    def _forecast_data(
        self,
        data: ModelData,
        steps: int = 1,
        forecast_origin=None,
        **kwargs,
    ) -> pd.DataFrame:
        """Produce each node's forecast bottom-up and return the root's.

        Each leaf is forecast once via its public ``forecast()`` (re-applying
        its own lag/dummy/formula config to the shared ``X``/``y``); each node
        is then evaluated in dependency order. ``**kwargs`` are forwarded to
        every leaf's ``forecast()``.
        """
        raw: dict[str, pd.DataFrame] = {}
        for leaf in self.spec.all_leaves():
            leaf_result = leaf._forecast_from_data(
                data,
                forecast_origin=forecast_origin,
                steps=steps,
                **kwargs,
            )
            raw[leaf.label] = leaf_result.forecast

        nodes = self.spec.nodes()
        for node in nodes:
            components = {name: raw[name] for name in node.child_names}
            transform = node.transform
            if isinstance(transform, ForecastModel):
                future = ModelData.from_components(
                    data,
                    _labelled_components(node, components, data),
                    kind="conditioning",
                )
                node_data = transform._raw_data.with_conditioning(future)
                node_data = node_data.published_after(data, transform.last_y_fit_date)
                transform_result = transform._forecast_from_data(
                    node_data,
                    forecast_origin=forecast_origin,
                    steps=steps,
                    **_node_transform_kwargs(kwargs, transform, fitted=True),
                )
                raw[node.name] = transform_result.forecast
            else:
                reduced = _reduce_components(components, self._node_targets[node.name])
                raw[node.name] = transform(reduced)

        node_names = {node.name for node in nodes}
        self.leaf_forecasts_ = {n: raw[n] for n in raw if n not in node_names}
        self.node_forecasts_ = {n: raw[n] for n in node_names}
        return raw[self.spec.name]
