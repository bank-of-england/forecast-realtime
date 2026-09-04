# ols_model.jl
# Plain ordinary least squares:  β = (XᵀX)⁻¹ Xᵀy
# Solved straight from the normal equations — no GLM / stats package.
# Toy example for testing the Julia integration; not for production use.
#
# Only defines fit() and forecast() — runner.jl handles CLI dispatch,
# parquet I/O and (de)serialisation.

using DataFrames
using LinearAlgebra

# `y` / `X` arrive from parquet with the pandas index in a `date` column;
# drop it and return a plain Float64 matrix.
_matrix(df) = Matrix{Float64}(df[!, filter(!=("date"), names(df))])

# ---- FIT ------------------------------------------------------------------
# Regress the single column of `y` on the columns of `X`.
function fit(y, X, params)
    X === nothing && error("This OLS model requires regressors X.")
    target = filter(!=("date"), names(y))[1]
    yv     = Float64.(y[!, target])
    Xm     = _matrix(X)

    beta = (Xm' * Xm) \ (Xm' * yv)          # β = (XᵀX)⁻¹ Xᵀy

    Dict("beta" => beta, "target" => target)
end

# ---- FORECAST -----------------------------------------------------------
# `X` holds the future regressor values (one row per step).
function forecast(model, steps, X, y, params)
    X === nothing && error("Future regressor values X (one row per step) are required.")
    Xf = _matrix(last(X, steps))
    DataFrame(model["target"] => Xf * model["beta"])
end
