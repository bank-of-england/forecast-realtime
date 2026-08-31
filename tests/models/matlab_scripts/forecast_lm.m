function result = forecast_lm(action, varargin)
% forecast_lm.m
% Ordinary least squares (OLS) regression on a time trend plus any regressors.

% fit:      result = forecast_lm('fit', y, X, params)
% forecast: result = forecast_lm('forecast', model, steps, X, y, params)

if strcmp(action, 'fit')
    y = drop_date(varargin{1});
    X = varargin{2};  % regressors (empty [] when there are none)

    n_obs  = height(y);
    values = double(y{:, 1});

    % Always include an intercept and a linear time trend.
    design     = [ones(n_obs, 1), (1:n_obs)'];
    regressors = {};
    if istable(X) && width(X) > 0
        X          = drop_date(X);
        regressors = X.Properties.VariableNames;
        design     = [design, double(X{1:n_obs, :})];
    end

    % Return a model struct — the runner saves it to model.mat
    % OLS via QR (numerically stable); solves X*coefficients = Y, i.e.
    % coefficients = (X'X)^-1 X'Y, where X = design and Y = values.
    result.coefficients = design \ values;
    result.n_obs        = n_obs;
    result.col_names    = y.Properties.VariableNames(1);
    result.regressors   = regressors;

elseif strcmp(action, 'forecast')
    model = varargin{1};
    steps = double(varargin{2});
    X     = varargin{3};  % future regressor values (one row per step)

    % Continue the time trend into the future.
    design = [ones(steps, 1), model.n_obs + (1:steps)'];

    if ~isempty(model.regressors)
        if ~istable(X) || height(X) < steps
            error('forecast_lm:MissingRegressors', ...
                ['Future regressor values (X) with one row per step ' ...
                'are required to forecast.']);
        end
        X      = drop_date(X);
        future = X(end - steps + 1:end, model.regressors);
        design = [design, double(future{:, :})];
    end

    % Return a table — the runner writes it to forecasts.parquet
    result = array2table(design * model.coefficients, ...
        'VariableNames', model.col_names);
else
    error('forecast_lm:UnknownAction', 'Unknown action: %s', action);
end
end

function tbl = drop_date(tbl)
% Remove the index column written by forecast-realtime, if present.
if any(strcmp(tbl.Properties.VariableNames, 'date'))
    tbl = removevars(tbl, 'date');
end
end
