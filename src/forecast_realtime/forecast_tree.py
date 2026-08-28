"""Nested forecast trees.

A ``TreeNode`` describes one node of a tree used to produce forecasts. Leaves
are bare ``ForecastModel`` instances (identified by their ``label``); internal
nodes are nested ``TreeNode`` instances (identified by their ``name``). Each
node stores a ``transform`` that produces its output from its direct
``children``. ``ForecastTree`` fits the whole tree and returns the root's
output.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from forecast_realtime.data_transformation import (
    DataTransformationPipeline,
    FittedDataTransformation,
    _validate_mapping_coverage,
)
from forecast_realtime.forecast_model import (
    FittedModelConfiguration,
    ForecastContext,
    ForecastModel,
    ForecastResult,
)

TransformType = Callable[[dict[str, pd.DataFrame]], pd.DataFrame] | ForecastModel

# Raw-input settings do not apply to synthetic child-output columns.
_NODE_RAW_INPUT_KWARGS = ("X_imputation", "input_frequencies")


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


def _strip_leaf_structure_kwargs(kwargs: dict) -> dict:
    """Return keyword arguments accepted by a node transform."""
    blocked = {"y_lags", "X_lags", "dummies"}
    return {k: v for k, v in kwargs.items() if k not in blocked}


def _node_transform_kwargs(kwargs: dict, transform: ForecastModel) -> dict:
    """Return keyword arguments for a stacking model transform."""
    node_kwargs = _strip_leaf_structure_kwargs(kwargs)
    for key in _NODE_RAW_INPUT_KWARGS:
        node_kwargs.pop(key, None)
    node_kwargs["data_transformation"] = None
    if transform.data_transformation is None:
        node_kwargs.pop("frequency", None)
        node_kwargs.pop("drop_transformation_nans", None)
    return node_kwargs


def _component_metric_mapping(component: ForecastModel) -> dict[str, str]:
    """Return the native output metric implied by a fitted component."""
    columns = list(getattr(component, "y", pd.DataFrame()).columns)
    if not columns:
        return {}
    return component.native_metric_mapping(target_variables=columns)


def _component_input_metrics(node: TreeNode) -> dict[str, str]:
    """Map a node's synthetic columns to its children's output metrics."""
    metrics = {}
    for child in node.children:
        if isinstance(child, ForecastModel):
            output = _component_metric_mapping(child)
            if output:
                metrics[child.label] = next(iter(output.values()))
        else:
            output = _component_metric_mapping(child.transform)
            if output:
                metrics[child.name] = next(iter(output.values()))
    return metrics


def _validate_node_transform_mapping(
    node: TreeNode,
    transform: ForecastModel,
    y_variables: list[str],
    X_variables: list[str],
) -> None:
    """Validate a node-owned mapping against the node's actual inputs."""
    mapping = transform.data_transformation
    if mapping is not None:
        formula = getattr(transform, "_formula", None)
        if formula is not None:
            y_variables = list(formula.y_cols)
            if not formula.has_wildcard:
                X_variables = [
                    variable for variable in X_variables if variable in formula.X_cols
                ]
        _validate_mapping_coverage(
            mapping,
            y_variables,
            X_variables,
            context=f"node {node.name!r}",
        )


def _concat_components(components: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate named per-source frames into one design matrix.

    A single-column source keeps that source's name as its column; a
    multi-column source has its columns prefixed with its name, avoiding
    collisions across sources.
    """
    frames = []
    for name, frame in components.items():
        if frame.shape[1] == 1:
            frames.append(frame.set_axis([name], axis=1))
        else:
            frames.append(frame.add_prefix(f"{name}_"))
    return pd.concat(frames, axis=1)


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

    def input_metric_requirements(
        self,
        y_variables: list[str],
        X_variables: list[str] | None = None,
        data_transformation=None,
    ) -> dict[str, tuple[str, ...]]:
        """Report raw metric requirements without flattening component plans.

        The returned tuples preserve contradictory leaf requirements. Realtime
        source selection may retain levels when they can derive every request;
        each fitted component still receives and executes its own plan.
        """
        fallback = self.data_transformation or data_transformation or {}
        requirements: dict[str, set[str]] = {variable: set() for variable in y_variables}
        requirements.update({variable: set() for variable in (X_variables or [])})

        def collect(component, owner_mapping):
            if isinstance(component, ForecastTree):
                nested_owner = (
                    component.data_transformation
                    if component.data_transformation is not None
                    else owner_mapping
                )
                for child in component.spec.children:
                    collect(child, nested_owner)
                if isinstance(component.spec.transform, ForecastModel):
                    collect(component.spec.transform, nested_owner)
                return

            mapping = component.data_transformation or owner_mapping
            if isinstance(component, ForecastModel):
                formula = getattr(component, "_formula", None)
                component_y_variables = (
                    list(formula.y_cols) if formula is not None else y_variables
                )
                component_X_variables = (
                    (
                        [
                            variable
                            for variable in (X_variables or [])
                            if variable in formula.X_cols
                        ]
                        if not formula.has_wildcard
                        else X_variables or []
                    )
                    if formula is not None
                    else X_variables or []
                )
                for variable in component_y_variables:
                    if variable in mapping:
                        requirements[variable].add(mapping[variable])
                for variable in component_X_variables:
                    if variable in mapping:
                        requirements[variable].add(mapping[variable])

        for leaf in self.spec.all_leaves():
            collect(leaf, fallback)
        for node in self.spec.nodes():
            if isinstance(node.transform, ForecastModel):
                collect(node.transform, fallback)

        return {
            variable: tuple(sorted(metrics))
            for variable, metrics in requirements.items()
            if metrics
        }

    required_input_metrics = input_metric_requirements

    def _resolve_child_data_transformation(self, kwargs: dict) -> dict:
        """Override the ``data_transformation`` fallback forwarded to children.

        When this tree owns a ``data_transformation``, its mapping
        becomes the ``data_transformation`` fallback forwarded to every leaf
        and nested tree that has no pipeline of its own (a leaf's/nested
        tree's own pipeline still wins); otherwise the call-level
        ``data_transformation`` already in ``kwargs`` is left untouched.
        """
        if self.data_transformation is None:
            return kwargs
        return {**kwargs, "data_transformation": self.data_transformation}

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

    def fit(self, y: pd.DataFrame, X: pd.DataFrame | None = None, **kwargs):
        """Fit the tree and return ``self`` after all nodes succeed."""
        candidate = copy.deepcopy(self)
        candidate._fit_impl(y, X, **kwargs)
        self._commit_fit(candidate)
        return self

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

    def _fit_impl(self, y: pd.DataFrame, X: pd.DataFrame | None = None, **kwargs):
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
        if not isinstance(y, pd.DataFrame):
            raise TypeError("y must be a pandas DataFrame")

        self._raw_y_history = y.copy()
        self._raw_X_history = X.copy() if X is not None else None

        self._refresh_capability_flags()
        kwargs = self._resolve_child_data_transformation(kwargs)
        self._fit(y, X, **kwargs)

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
                None,
                y_variables=list(self.y.columns),
                X_variables=None,
                y_frequencies={},
                X_frequencies={},
                frequency=None,
                X_imputation=None,
                pipeline_source="none",
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

    def forecast(
        self,
        steps: int = 1,
        X: pd.DataFrame | None = None,
        y: pd.DataFrame | None = None,
        decomp: bool = False,
        context: ForecastContext | None = None,
        **kwargs,
    ) -> ForecastResult:
        """Produce each node's forecast bottom-up and return the root's.

        Overrides the base ``ForecastModel.forecast()`` so the root ``y``/``X``
        (history plus any conditioning/future paths) are forwarded to every
        leaf/nested tree raw (untransformed): each leaf/nested tree combines
        them with its own raw fit history and applies its own transformation
        exactly once, inside its own ``forecast()``. ``**kwargs`` are
        forwarded to every leaf's ``forecast()``, with ``data_transformation``
        overridden by this tree's own pipeline when it has one.
        """
        if context is None:
            context = ForecastContext(
                y_history=self._raw_y_history,
                X_history=self._raw_X_history,
                y_conditioning=y,
                X_conditioning=X,
                forecast_origin=self.last_y_fit_date,
            )
        return self.predict(
            context,
            steps=steps,
            decomp=decomp,
            **kwargs,
        )

    def predict(
        self,
        context: ForecastContext,
        steps: int = 1,
        decomp: bool = False,
        **kwargs,
    ) -> ForecastResult:
        """Evaluate the tree from an explicit, immutable forecast context."""
        if not isinstance(steps, int) or steps <= 0:
            raise ValueError("'Steps' must be an integer greater than zero")

        kwargs = self._resolve_child_data_transformation(kwargs)
        forecast_origin = context.forecast_origin or context.y_history.index[-1]
        forecast = self._forecast(
            context=context,
            steps=steps,
            X=context.X_conditioning,
            y=context.y_conditioning,
            forecast_origin=forecast_origin,
            **kwargs,
        )
        return self._finalise_forecast(
            forecast,
            steps=steps,
            forecast_origin=forecast_origin,
            decomp=decomp,
            decomp_kwargs={
                "X": context.X_conditioning,
                "y": context.y_conditioning,
                "forecast_origin": forecast_origin,
                **kwargs,
            },
        )

    def _fit(self, y: pd.DataFrame, X: pd.DataFrame | None = None, **kwargs):
        for leaf in self.spec.all_leaves():
            leaf_y = y
            leaf_formula = getattr(leaf, "_formula", None)
            if leaf_formula:
                target_rows = leaf_formula.extract_y(y).notna().all(axis=1)
                leaf_y = y.loc[target_rows]
            leaf.fit(leaf_y, X, **kwargs)

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
                X_node = _concat_components(components)
                X_node = X_node.dropna(how="all")
                common_index = X_node.index.intersection(y.index)
                _validate_node_transform_mapping(
                    node, transform, list(y.columns), list(X_node.columns)
                )
                transform.fit(
                    y.loc[common_index],
                    X_node.loc[common_index],
                    **self._node_fit_kwargs(
                        kwargs,
                        transform,
                        list(y.columns),
                        _component_input_metrics(node),
                    ),
                )
                raw[node.name] = _as_frame(transform.fitted_values)
            else:
                reduced = _reduce_components(components, self._node_targets[node.name])
                raw[node.name] = transform(reduced)

        self.fitted_values_ = raw[self.spec.name]
        return self

    @staticmethod
    def _node_fit_kwargs(
        kwargs: dict,
        transform: ForecastModel,
        y_variables: list[str],
        X_input_metrics: dict[str, str],
    ) -> dict:
        """Prepare a stacking fit call with synthetic source provenance."""
        node_kwargs = _node_transform_kwargs(kwargs, transform)
        y_input_metrics = node_kwargs.pop("y_input_metrics", None) or {}
        node_kwargs.pop("X_input_metrics", None)
        if transform.data_transformation is not None:
            mapping = transform.data_transformation
            node_kwargs["y_input_metrics"] = {
                variable: y_input_metrics.get(variable, "levels")
                for variable in y_variables
                if variable in mapping
            }
            node_kwargs["X_input_metrics"] = X_input_metrics
        return node_kwargs

    def _forecast(
        self,
        context: ForecastContext,
        steps: int = 1,
        X: pd.DataFrame | None = None,
        y: pd.DataFrame | None = None,
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
            leaf_result = leaf.forecast(
                context=ForecastContext(
                    y_history=context.y_history,
                    X_history=context.X_history,
                    y_conditioning=context.y_conditioning,
                    X_conditioning=context.X_conditioning,
                    forecast_origin=forecast_origin,
                    y_conditioning_input_metrics=context.y_conditioning_input_metrics,
                    X_conditioning_input_metrics=context.X_conditioning_input_metrics,
                ),
                steps=steps,
                **kwargs,
            )
            raw[leaf.label] = leaf_result.forecast

        nodes = self.spec.nodes()
        for node in nodes:
            components = {name: raw[name] for name in node.child_names}
            transform = node.transform
            if isinstance(transform, ForecastModel):
                X_node = _concat_components(components)
                ragged_y = context.y_history.loc[
                    context.y_history.index > transform.last_y_fit_date
                ]
                node_y_conditioning = (
                    context.y_conditioning.combine_first(ragged_y)
                    if context.y_conditioning is not None
                    else ragged_y
                )
                if node_y_conditioning.empty:
                    node_y_conditioning = None
                _validate_node_transform_mapping(
                    node,
                    transform,
                    list(context.y_history.columns),
                    list(X_node.columns),
                )
                transform_result = transform.forecast(
                    context=ForecastContext(
                        y_history=transform._raw_y_history,
                        X_history=transform._raw_X_history,
                        y_conditioning=node_y_conditioning,
                        X_conditioning=X_node,
                        forecast_origin=forecast_origin,
                        X_conditioning_input_metrics=_component_input_metrics(node),
                    ),
                    steps=steps,
                    **_node_transform_kwargs(kwargs, transform),
                )
                raw[node.name] = transform_result.forecast
            else:
                reduced = _reduce_components(components, self._node_targets[node.name])
                raw[node.name] = transform(reduced)

        node_names = {node.name for node in nodes}
        self.leaf_forecasts_ = {n: raw[n] for n in raw if n not in node_names}
        self.node_forecasts_ = {n: raw[n] for n in node_names}
        return raw[self.spec.name]
