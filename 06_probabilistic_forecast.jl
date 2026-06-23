##############################################################################
# 06_probabilistic_forecast.jl  —  Pronóstico probabilístico y verificación
#
# Metodología:
#   - Combina incertidumbre de parámetros (posterior BUDE) e incertidumbre
#     estocástica (ruido SDE heterocedástico) para obtener la distribución
#     predictiva posterior conjunta.
#   - Pronósticos rodantes H = 1..12 meses con M = 1000 miembros.
#   - Métricas probabilísticas:
#       1. CRPS (score continuo de probabilidad por rangos)
#       2. Cobertura del intervalo de predicción al 90%
#       3. Score de Brier (evento El Niño: SST > 0.5 °C)
#
# Salidas:
#   figures/06_probabilistic_predictions.png
#   figures/06_verification_scores.png
##############################################################################

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using DataFrames, CSV, Dates, JLD2, JSON3
using Lux, CairoMakie, ComponentArrays
using Random, Statistics, LinearAlgebra, SpecialFunctions
using DifferentialEquations

cd(@__DIR__)
mkpath("figures")

isdefined(Main, :EnsoModels) || include("src/EnsoModels.jl")
using .EnsoModels

const rng = Xoshiro(42)
const Δt  = EnsoModels.Δt_default

##############################################################################
# 1. Cargar datos
##############################################################################

df = CSV.read("data/enso_dataset.csv", DataFrame)

norm_params = JSON3.read(read("data/normalization.json", String))
T_scale     = Float32(norm_params["T_scale"])
h_scale     = Float32(norm_params["h_scale"])

mask_train = (year.(df.date) .>= 1980) .& (year.(df.date) .<= 2010)
df_train   = df[mask_train, :]
T_train    = Float32.(df_train.T_anomaly)

mask_test  = (year.(df.date) .>= 2011) .& (year.(df.date) .<= 2021)
df_test    = df[mask_test, :]
T_test     = Float32.(df_test.T_anomaly)
h_test     = Float32.(df_test.h_anomaly)
dates_test = df_test.date
t_test     = Float32.(year.(dates_test) .+ (month.(dates_test) .- 1) ./ 12)
N_test     = length(T_test)

@info "Ventana de test: $(dates_test[1]) – $(dates_test[end])  (N=$N_test)"

##############################################################################
# 2. Cargar modelos
##############################################################################

θ_bayes = load("data/ude_bayesian_params.jld2", "theta_bayes")
sde_par = load("data/sde_noise_params.jld2")

γ_T = Float32.(sde_par["γ_T"])
γ_h = Float32.(sde_par["γ_h"])

snn_f, snn_g, ps_f, ps_g = setup_bude_networks(rng)

##############################################################################
# 3. Pronósticos rodantes probabilísticos (H = 1..12)
##############################################################################

H_max = 12
M     = 1000

crps_busde  = zeros(H_max)
crps_clima  = zeros(H_max)
brier_busde = zeros(H_max)
coverage_90 = zeros(H_max)

forecast_probs_6m  = Float32[]
observed_events_6m = Float32[]

@info "Calculando pronósticos rodantes probabilísticos con $M realizaciones..."
for H in 1:H_max
    crps_list   = Float64[]
    crps_c_list = Float64[]
    brier_list  = Float64[]
    cov_count   = 0
    n_valid     = N_test - H

    θ_samples = [θ_bayes.μ + exp.(θ_bayes.ρ) .* ComponentArray(randn(rng, Float32, length(θ_bayes.μ)), getaxes(θ_bayes.μ)) for m in 1:M]

    discrete_step_busde(u, p, t) = Float32[busde_step(p[1], u[1], u[2], p[2], snn_f, snn_g, T_scale, h_scale, γ_T, γ_h; Δt=Δt)...]

    prob_sde = DiscreteProblem(discrete_step_busde, Float32[0.0, 0.0], (0, H), (Xoshiro(42), θ_bayes.μ))

    function prob_func_sde(prob, idx, repeat)
        m = (idx - 1) % M + 1
        i = (idx - 1) ÷ M + 1
        traj_rng = Xoshiro(42 + idx + H * 100000)
        remake(prob, u0=Float32[T_test[i], h_test[i]], p=(traj_rng, θ_samples[m]))
    end

    ensemble_prob_sde = EnsembleProblem(prob_sde; prob_func=prob_func_sde)
    sim_sde = solve(ensemble_prob_sde, FunctionMap(), EnsembleThreads(); trajectories=M*n_valid)

    for i in 1:n_valid
        y_true = T_test[i + H]
        pred_T = zeros(Float32, M)
        for m in 1:M
            idx = (i - 1) * M + m
            pred_T[m] = sim_sde.u[idx].u[end][1]
        end

        push!(crps_list,   crps_ensemble(pred_T, y_true))
        push!(crps_c_list, crps_climatology(y_true, T_train))

        p_elnino = mean(pred_T .> 0.5f0)
        o_elnino = y_true > 0.5f0 ? 1.0f0 : 0.0f0
        push!(brier_list, (p_elnino - o_elnino)^2)

        q05 = quantile(pred_T, 0.05f0)
        q95 = quantile(pred_T, 0.95f0)
        if q05 <= y_true <= q95; cov_count += 1; end

        if H == 6
            push!(forecast_probs_6m,  p_elnino)
            push!(observed_events_6m, o_elnino)
        end
    end

    crps_busde[H]  = mean(crps_list)
    crps_clima[H]  = mean(crps_c_list)
    brier_busde[H] = mean(brier_list)
    coverage_90[H] = cov_count / n_valid
end

@info "===== Scores de habilidad probabilísticos ====="
for H in [1, 3, 6, 12]
    @info "  H=$H: CRPS BUSDE = $(round(crps_busde[H], digits=3)) °C, Climatología = $(round(crps_clima[H], digits=3)) °C"
    @info "        Score de Brier = $(round(brier_busde[H], digits=3)), Cobertura 90% = $(round(coverage_90[H]*100, digits=1))%"
end

##############################################################################
# 4. Pronóstico rodante denso a 6 meses (visualización completa 2011–2021)
##############################################################################

H_plot  = 6
M_plot  = 500
S_plot  = 3   # inicializar un nuevo pronóstico cada 3 meses

T_mean_rolling = fill(NaN32, N_test)
T_q10_rolling  = fill(NaN32, N_test)
T_q90_rolling  = fill(NaN32, N_test)

@info "Construyendo la visualización del pronóstico rodante denso..."

θ_samples_plot = [θ_bayes.μ + exp.(θ_bayes.ρ) .* ComponentArray(randn(rng, Float32, length(θ_bayes.μ)), getaxes(θ_bayes.μ)) for m in 1:M_plot]

starts_plot   = collect(1:S_plot:(N_test - H_plot))
N_starts_plot = length(starts_plot)

discrete_step_busde_plot(u, p, t) = Float32[busde_step(p[1], u[1], u[2], p[2], snn_f, snn_g, T_scale, h_scale, γ_T, γ_h; Δt=Δt)...]

prob_sde_plot = DiscreteProblem(discrete_step_busde_plot, Float32[0.0, 0.0], (0, H_plot), (Xoshiro(42), θ_bayes.μ))

function prob_func_plot(prob, idx, repeat)
    m     = (idx - 1) % M_plot + 1
    s_idx = (idx - 1) ÷ M_plot + 1
    s     = starts_plot[s_idx]
    remake(prob, u0=Float32[T_test[s], h_test[s]], p=(Xoshiro(100 + idx), θ_samples_plot[m]))
end

ensemble_prob_plot = EnsembleProblem(prob_sde_plot; prob_func=prob_func_plot)
sim_plot = solve(ensemble_prob_plot, FunctionMap(), EnsembleThreads(); trajectories=M_plot*N_starts_plot)

for (s_idx, s) in enumerate(starts_plot)
    seg_T = zeros(Float32, H_plot + 1, M_plot)
    for m in 1:M_plot
        idx = (s_idx - 1) * M_plot + m
        for k in 0:H_plot
            seg_T[k + 1, m] = sim_plot.u[idx].u[k + 1][1]
        end
    end
    for k in 0:H_plot
        idx = s + k
        if idx <= N_test
            T_mean_rolling[idx] = mean(seg_T[k + 1, :])
            T_q10_rolling[idx]  = quantile(seg_T[k + 1, :], 0.10f0)
            T_q90_rolling[idx]  = quantile(seg_T[k + 1, :], 0.90f0)
        end
    end
end

##############################################################################
# 5. Figuras
##############################################################################

const Figure = CairoMakie.Figure
const Axis   = CairoMakie.Axis

# Figura 1: envolventes del pronóstico rodante denso
fig1 = Figure(size=(1400, 900), fontsize=13)

axA = Axis(fig1[1, 1:2],
    xlabel = "Año",
    ylabel = "Anomalía de SST (°C)",
    title  = "SST en Niño 3.4 — Pronósticos probabilísticos rodantes a 6 meses de BUSDE (2011–2021)\n" *
             "Envolvente: percentiles 10–90  |  Media actualizada cada $S_plot meses")

band!(axA, t_test, T_q10_rolling, T_q90_rolling; color=(:darkorange, 0.20), label="Envolvente 10–90%")
lines!(axA, t_test, T_test;         color=:firebrick,  linewidth=2.0, label="Observado")
lines!(axA, t_test, T_mean_rolling; color=:darkorange, linewidth=1.8, label="Media del ensemble")
band!(axA, t_test, fill(0.5f0, N_test), fill(4.0f0, N_test); color=(:red, 0.05))
hlines!(axA, [0.5f0];  color=:red,  linestyle=:dot, linewidth=1.0)
hlines!(axA, [-0.5f0]; color=:blue, linestyle=:dot, linewidth=1.0)
axislegend(axA, position=:lt, labelsize=11)

valid_roll = .!isnan.(T_mean_rolling)
axB = Axis(fig1[2, 1],
    xlabel = "Media del ensemble BUSDE (°C)",
    ylabel = "Observado (°C)",
    title  = "Media del pronóstico vs Observado (H = 1–$H_plot meses)")
scatter!(axB, T_mean_rolling[valid_roll], T_test[valid_roll]; color=:darkorange, markersize=4, alpha=0.4)
lines!(axB, [-3f0, 3f0], [-3f0, 3f0]; color=:gray, linestyle=:dash, linewidth=1.5)

r_roll    = cor(T_mean_rolling[valid_roll], T_test[valid_roll])
rmse_roll = sqrt(mean((T_mean_rolling[valid_roll] .- T_test[valid_roll]).^2))
text!(axB, -2.5f0, 2.2f0;
      text="r = $(round(r_roll, digits=2))\nRMSE = $(round(rmse_roll, digits=2)) °C",
      fontsize=11, color=:black)

cov_by_month = [T_q10_rolling[i] <= T_test[i] <= T_q90_rolling[i]
                for i in 1:N_test if !isnan(T_q10_rolling[i])]
t_valid = [t_test[i] for i in 1:N_test if !isnan(T_q10_rolling[i])]
axC = Axis(fig1[2, 2],
    xlabel = "Año",
    ylabel = "Dentro del intervalo del 90%",
    title  = "Cobertura puntual del 90% (general: $(round(mean(cov_by_month)*100, digits=1))%, ideal: 90%)")
barplot!(axC, t_valid, Float32.(cov_by_month);
      color=[c ? (:darkorange, 0.7) : (:firebrick, 0.6) for c in cov_by_month],
      width=0.09)
hlines!(axC, [0.9f0]; color=:gray, linestyle=:dash, linewidth=1.5)

save("figures/06_probabilistic_predictions.png", fig1; px_per_unit=2)
@info "Guardado figures/06_probabilistic_predictions.png"

# Figura 2: scores de verificación
fig2 = Figure(size=(1050, 550), fontsize=13)
Hs = 1:H_max

ax2a = Axis(fig2[1, 1],
    xlabel="Tiempo de anticipación (meses)",
    ylabel="CRPS (°C)",
    title="CRPS vs Tiempo de anticipación")
lines!(ax2a, Hs, crps_clima; color=:gray,      linewidth=1.5, linestyle=:dash, label="Climatología")
lines!(ax2a, Hs, crps_busde; color=:darkorange, linewidth=2.0, label="BUSDE")
scatter!(ax2a, Hs, crps_busde; color=:darkorange, markersize=6)
axislegend(ax2a, position=:lt)

skill = 1.0 .- crps_busde ./ crps_clima
ax2b = Axis(fig2[1, 2],
    xlabel="Tiempo de anticipación (meses)",
    ylabel="CRPSS (mayor = mejor)",
    title="Score de habilidad CRPS vs Climatología")
hlines!(ax2b, [0.0]; color=:gray, linestyle=:dash, linewidth=1.0)
barplot!(ax2b, collect(Float64.(Hs)), skill;
      color=[s >= 0 ? (:darkorange, 0.75) : (:firebrick, 0.6) for s in skill],
      width=0.7)

ax2c = Axis(fig2[1, 3],
    xlabel="Tiempo de anticipación (meses)",
    ylabel="Cobertura de PI del 90%",
    title="Cobertura de intervalo (ideal = 0.90)")
hlines!(ax2c, [0.90]; color=:gray, linestyle=:dash, linewidth=1.5, label="Ideal")
lines!(ax2c, Hs, coverage_90; color=:darkorange, linewidth=2.0, label="BUSDE")
scatter!(ax2c, Hs, coverage_90; color=:darkorange, markersize=6)
axislegend(ax2c, position=:lt)

save("figures/06_verification_scores.png", fig2; px_per_unit=2)
@info "Guardado figures/06_verification_scores.png"

@info "===== Verificación completada ====="
