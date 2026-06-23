##############################################################################
# 08_crossval_rolling.jl  -  Validación cruzada rolling-origin (ENSO)
#
# Metodología:
#   - Esquema rolling-origin con 5 folds en ventanas crecientes:
#       Fold 1: Test 1995–1999
#       Fold 2: Test 2000–2004
#       Fold 3: Test 2005–2009
#       Fold 4: Test 2010–2014
#       Fold 5: Test 2015–2019
#   - Evalúa la media de la posterior θ.μ en modo fuera de muestra.
#   - Compara UDE vs. OLS vs. climatología en H = 1, 3, 6 meses.
#
# Salidas:
#   figures/08_crossval_rolling.png
#   data/crossval_metrics.json
##############################################################################

using Pkg
Pkg.activate(@__DIR__)

using DataFrames, CSV, Dates, JSON3, JLD2
using Lux, CairoMakie, ComponentArrays
using Random, Statistics, LinearAlgebra

cd(@__DIR__)
mkpath("data")
mkpath("figures")

isdefined(Main, :EnsoModels) || include("src/EnsoModels.jl")
using .EnsoModels

const rng = Xoshiro(42)

norm_params = JSON3.read(read("data/normalization.json", String))
T_scale     = Float32(norm_params["T_scale"])
h_scale     = Float32(norm_params["h_scale"])

p_base = JSON3.read(read("data/baseline_params.json", String))
a_ols  = Float32(p_base["a"])
b_ols  = Float32(p_base["b"])
c_ols  = Float32(p_base["c"])

θ = load("data/ude_bayesian_params.jld2", "theta_bayes")
snn_f, snn_g, ps_f, ps_g = setup_bude_networks(rng)

df = CSV.read("data/enso_dataset.csv", DataFrame)

folds = [
    (name="Fold 1", start_year=1995, end_year=1999),
    (name="Fold 2", start_year=2000, end_year=2004),
    (name="Fold 3", start_year=2005, end_year=2009),
    (name="Fold 4", start_year=2010, end_year=2014),
    (name="Fold 5", start_year=2015, end_year=2019)
]

lead_times = [1, 3, 6]
results    = Dict()

metrics(ŷ, y) = begin
    r = Float32(sqrt(mean((ŷ .- y).^2)))
    a = Float32(cor(ŷ, y))
    (rmse = r, acc = isnan(a) ? 0.0f0 : a)
end

@info "Comenzando validación cruzada rolling-origin (5 folds, H ∈ {1, 3, 6})..."

for fold in folds
    @info "Procesando $(fold.name) (Test: $(fold.start_year) – $(fold.end_year))..."

    mask_test = (year.(df.date) .>= fold.start_year) .& (year.(df.date) .<= fold.end_year)
    df_test   = df[mask_test, :]
    T_test    = Float32.(df_test.T_anomaly)
    h_test    = Float32.(df_test.h_anomaly)
    N_test    = length(T_test)

    fold_results = Dict()

    for H in lead_times
        obs_H      = Float32[]
        pred_ude   = Float32[]
        pred_ols   = Float32[]
        pred_clim  = Float32[]

        for i in 1:(N_test - H)
            push!(pred_clim, 0.0f0)

            T_o, h_o = T_test[i], h_test[i]
            for _ in 1:H
                T_o, h_o = euler_step_ols(T_o, h_o, a_ols, b_ols, c_ols; Δt=EnsoModels.Δt_default)
            end
            push!(pred_ols, T_o)

            T_u, h_u = T_test[i], h_test[i]
            for _ in 1:H
                T_u, h_u = euler_step_bude(
                    T_u, h_u, θ.μ, snn_f, snn_g, T_scale, h_scale; Δt=EnsoModels.Δt_default
                )
            end
            push!(pred_ude, T_u)

            push!(obs_H, T_test[i + H])
        end

        m_clim = metrics(pred_clim, obs_H)
        m_ols  = metrics(pred_ols,  obs_H)
        m_ude  = metrics(pred_ude,  obs_H)

        fold_results[string(H)] = Dict(
            "climatology" => Dict("rmse" => m_clim.rmse, "acc" => m_clim.acc),
            "ols"         => Dict("rmse" => m_ols.rmse,  "acc" => m_ols.acc),
            "ude"         => Dict("rmse" => m_ude.rmse,  "acc" => m_ude.acc)
        )
    end
    results[fold.name] = fold_results
end

open("data/crossval_metrics.json", "w") do io
    JSON3.write(io, results)
end
@info "Métricas guardadas en data/crossval_metrics.json"

println("\n" * "="^70)
println("RESUMEN DE VALIDACIÓN CRUZADA ROLLING-ORIGIN (RMSE)")
println("="^70)
println(string(rpad("Fold", 12), " | Lead", " | Climatología", " | OLS Baseline ", " | UDE Model    "))
println("-"^70)
for fold in folds
    for H in lead_times
        res       = results[fold.name][string(H)]
        clim_rmse = round(res["climatology"]["rmse"], digits=3)
        ols_rmse  = round(res["ols"]["rmse"], digits=3)
        ude_rmse  = round(res["ude"]["rmse"], digits=3)
        println(string(
            rpad(fold.name, 12), " | ",
            rpad(string(H) * "m", 4), " | ",
            rpad(string(clim_rmse) * " °C", 12), " | ",
            rpad(string(ols_rmse)  * " °C", 13), " | ",
            rpad(string(ude_rmse)  * " °C", 12)
        ))
    end
    println("-"^70)
end

##############################################################################
# Graficar comparativa
##############################################################################

@info "Generando gráficos de validación cruzada..."
const Figure = CairoMakie.Figure
const Axis   = CairoMakie.Axis

fig = Figure(size=(1200, 800), fontsize=13)

fold_names  = [f.name for f in folds]
x_positions = 1:5

for (i_H, H) in enumerate(lead_times)
    ax = Axis(fig[i_H, 1],
              xticks=(x_positions, fold_names),
              ylabel="RMSE (°C)",
              title="Habilidad de pronóstico a $H meses")

    clim_rmse = [results[f.name][string(H)]["climatology"]["rmse"] for f in folds]
    ols_rmse  = [results[f.name][string(H)]["ols"]["rmse"] for f in folds]
    ude_rmse  = [results[f.name][string(H)]["ude"]["rmse"] for f in folds]

    scatterlines!(ax, x_positions, clim_rmse; color=:black,      linestyle=:dot,  markersize=8, linewidth=1.5, label="Climatología")
    scatterlines!(ax, x_positions, ols_rmse;  color=:gray,       linestyle=:dash, markersize=8, linewidth=1.5, label="OLS")
    scatterlines!(ax, x_positions, ude_rmse;  color=:darkorange,                  markersize=10, linewidth=2.5, label="BUSDE Media")

    if i_H == 1
        axislegend(ax, position=:rt, framevisible=false)
    end
end

save("figures/08_crossval_rolling.png", fig; px_per_unit=2)
@info "Figura guardada en figures/08_crossval_rolling.png"
@info "===== Validación cruzada completada ====="
