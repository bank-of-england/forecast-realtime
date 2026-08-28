% runner.m — shipped with forecast_realtime
% Handles parameter reading, data loading, model serialisation, and forecast output.
% The user function only defines:
%   my_func('fit', y, X, params)              → returns a model struct
%   my_func('forecast', model, steps, X, y, params) → returns a table
%
% Called by Python as:
%   matlab -batch "runner('<user_func>', 'fit', '<cache_dir>', '<params_path>')"
%   matlab -batch "runner('<user_func>', 'forecast', '<cache_dir>', <steps>, '<params_path>')"

function runner(user_function, action, cache_dir, steps_or_params, params_path)

    % Load y from cache_dir
    y = parquetread(fullfile(cache_dir, 'y.parquet'));

    % Optional regressors — empty [] when the model was fit without any X.
    % At fit time these are the training regressors; at forecast time they
    % are the future regressor values (one row per step).
    x_path = fullfile(cache_dir, 'X.parquet');
    if isfile(x_path)
        X = parquetread(x_path);
    else
        X = [];
    end

    fh = str2func(user_function);

    if strcmp(action, 'fit')
        actual_params_path = steps_or_params;
        params = read_params(actual_params_path);

        model = fh('fit', y, X, params);
        save(fullfile(cache_dir, 'model.mat'), 'model');

    elseif strcmp(action, 'forecast')
        steps  = steps_or_params;
        params = read_params(params_path);

        loaded = load(fullfile(cache_dir, 'model.mat'), 'model');
        result = fh('forecast', loaded.model, steps, X, y, params);
        parquetwrite(fullfile(cache_dir, 'forecasts.parquet'), result);
    else
        error('Unknown action: %s', action);
    end
end

function params = read_params(params_path)
    if isfile(params_path)
        params = table2struct(parquetread(params_path));
    else
        params = struct();
    end
end
