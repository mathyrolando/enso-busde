##############################################################################
# 04_evaluation.jl  —  Evaluación fuera de muestra (2011–2021)
#
# Metodología:
#   - Carga la distribución posterior de parámetros desde 03_ude.jl.
#   - Simulación ensemble a largo plazo desde la condición inicial de test.
#   - Pronósticos rodantes en horizontes H = 1..18 meses (media del ensemble).
#   - Gráfico espagueti a 6 meses para visualizar incertidumbre epistémica.
#
# Salidas:
#   figures/04_test_rollout_bayesian.png
#   figures/04_skill_score_bayesian.png
#   figures/04_rolling_6m_bayesian.png
##############################################################################

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using DataFrames, CSV, Dates, JSON3, JLD2
using Lux, CairoMakie, ComponentArrays
using Random, Statistics, LinearAlgebra
using DifferentialEquations

cd(@__DIR__)
mkpath("figures")

isdefined(Main, :EnsoModels) || include("src/EnsoModels.jl")
using .EnsoModels

const rng = Xoshiro(42)
const Δt  = EnsoModels.Δt_default

##############################################################################
# 1. Cargar datos y parámetros
##############################################################################

df = CSV.read("data/enso_dataset.csv", DataFrame)

norm_params = JSON3.read(read("data/normalization.json", String))
T_scale     = Float32(norm_params["T_scale"])
h_scale     = Float32(norm_params["h_scale"])

mask_test  = (year.(df.date) .>= 2011) .& (year.(df.date) .<= 2021)
df_test    = df[mask_test, :]
T_test     = Float32.(df_test.T_anomaly)
h_test     = Float32.(df_test.h_anomaly)
dates_test = df_test.date
t_test     = Float32.(year.(dates_test) .+ (month.(dates_test) .- 1) ./ 12)
N_test     = length(T_test)

@info "Ventana de test: $(dates_test[1]) – $(dates_test[end])  (N=$N_test)"

##############################################################################
# 2. Cargar parámetros y redes
##############################################################################

θ = load("data/ude_bayesian_params.jld2", "theta_bayes")

snn_f, snn_g, ps_f, ps_g = setup_bude_networks(rng)

p_base = JSON3.read("data/baseline_params.json")
a_ols  = Float32(p_base["a"]); b_ols = Float32(p_base["b"]); c_ols = Float32(p_base["c"])

a_lin  = θ.μ.a;  b_lin = exp(θ.μ.log_b);  c_lin = exp(θ.μ.log_c)

##############################################################################
# 3. Simulación ensemble a largo plazo
##############################################################################

function rollout_ensemble(T0, h0, n_steps, M=100)
    function discrete_step(u, p, t)
        T_next, h_next = euler_step_bude(u[1], u[2], p, snn_f, snn_g, T_scale, h_scale; Δt=Δt)
        return Float32[T_next, h_next]
    end

    u0    = Float32[T0, h0]
    tspan = (0, n_steps - 1)
    prob  = DiscreteProblem(discrete_step, u0, tspan, θ.μ)

    function prob_func(prob, i, repeat)
        ϵ_sample = ComponentArray(randn(rng, Float32, length(θ.μ)), getaxes(θ.μ))
        θ_sample = θ.μ + exp.(θ.ρ) .* ϵ_sample
        remake(prob, p=θ_sample)
    end

    ensemble_prob = EnsembleProblem(prob; prob_func=prob_func)
    sim = solve(ensemble_prob, FunctionMap(), EnsembleThreads(); trajectories=M)

    T_ens = zeros(Float32, n_steps, M)
    h_ens = zeros(Float32, n_steps, M)
    for m in 1:M, k in 1:n_steps
        T_ens[k, m] = sim.u[m].u[k][1]
        h_ens[k, m] = sim.u[m].u[k][2]
    end

    T_mean = [mean(T_ens[i, :]) for i in 1:n_steps]
    h_mean = [mean(h_ens[i, :]) for i in 1:n_steps]
    T_q10  = [quantile(T_ens[i, :], 0.1f0) for i in 1:n_steps]
    T_q90  = [quantile(T_ens[i, :], 0.9f0) for i in 1:n_steps]
    h_q10  = [quantile(h_ens[i, :], 0.1f0) for i in 1:n_steps]
    h_q90  = [quantile(h_ens[i, :], 0.9f0) for i in 1:n_steps]

    return T_mean, h_mean, T_q10, T_q90, h_q10, h_q90
end

function rollout_ols(a_p, b_p, c_p, T0, h0, n_steps)
    T_arr = zeros(Float32, n_steps)
    h_arr = zeros(Float32, n_steps)
    T_arr[1] = T0; h_arr[1] = h0
    T_t, h_t = T0, h0
    for k in 2:n_steps
        T_t, h_t  = euler_step_ols(T_t, h_t, a_p, b_p, c_p; Δt=Δt)
        T_arr[k] = T_t; h_arr[k] = h_t
    end
    return T_arr, h_arr
end

T_mean, h_mean, T_q10, T_q90, h_q10, h_q90 = rollout_ensemble(T_test[1], h_test[1], N_test, 100)
T_lin_long, h_lin_long = rollout_ols(a_lin, b_lin, c_lin, T_test[1], h_test[1], N_test)
T_ols_long, h_ols_long = rollout_ols(a_ols, b_ols, c_ols, T_test[1], h_test[1], N_test)

metrics(ŷ, y) = (sqrt(mean((ŷ .- y).^2)), cor(ŷ, y))
rmse_ude_t, r_ude_t = metrics(T_mean, T_test)
rmse_lin_t, r_lin_t = metrics(T_lin_long, T_test)
rmse_ols_t, r_ols_t = metrics(T_ols_long, T_test)

@info "Integrando con Tsit5 en evaluación..."
sol_tsit5 = solve_bude_ode(T_test[1], h_test[1], θ.μ, snn_f, snn_g, T_scale, h_scale, (0.0f0, Float32(N_test-1)*Δt); Δt=Δt)
T_tsit5 = [sol_tsit5.u[i][1] for i in 1:length(sol_tsit5.u)]
h_tsit5 = [sol_tsit5.u[i][2] for i in 1:length(sol_tsit5.u)]
n_min   = min(length(T_mean), length(T_tsit5))
rmse_tsit5_t, r_tsit5_t = metrics(T_tsit5[1:n_min], T_test[1:n_min])

@info "===== Simulación larga en test ====="
@info "  OLS:         RMSE=$(round(rmse_ols_t, digits=3)) °C, r=$(round(r_ols_t, digits=3))"
@info "  BUDE Lin.:   RMSE=$(round(rmse_lin_t, digits=3)) °C, r=$(round(r_lin_t, digits=3))"
@info "  BUDE Media:  RMSE=$(round(rmse_ude_t, digits=3)) °C, r=$(round(r_ude_t, digits=3))"
@info "  BUDE Tsit5:  RMSE=$(round(rmse_tsit5_t, digits=3)) °C, r=$(round(r_tsit5_t, digits=3))"
@info "  Diferencia máxima EC vs Tsit5: $(round(maximum(abs.(T_mean[1:n_min] .- T_tsit5[1:n_min])), digits=5)) °C"

##############################################################################
# 4. Pronósticos rodantes (H = 1..18 meses)
##############################################################################

H_max = 18
M     = 50

rmse_ude_h  = zeros(H_max); r_ude_h  = zeros(H_max)
rmse_lin_h  = zeros(H_max); r_lin_h  = zeros(H_max)
rmse_ols_h  = zeros(H_max); r_ols_h  = zeros(H_max)
rmse_pers_h = zeros(H_max); r_pers_h = zeros(H_max)
rmse_clim_h = zeros(H_max); r_clim_h = zeros(H_max)

@info "Calculando skill scores rodantes (H = 1..$H_max meses)..."
for H in 1:H_max
    pred_lin  = Float32[]; pred_ols  = Float32[]
    pred_pers = Float32[]; pred_clim = Float32[]
    obs_H     = Float32[]

    for i in 1:(N_test - H)
        T_l, h_l = T_test[i], h_test[i]
        T_o, h_o = T_test[i], h_test[i]
        for _ in 1:H
            T_l, h_l = euler_step_ols(T_l, h_l, a_lin, b_lin, c_lin; Δt=Δt)
            T_o, h_o = euler_step_ols(T_o, h_o, a_ols, b_ols, c_ols; Δt=Δt)
        end
        push!(pred_lin, T_l)
        push!(pred_ols, T_o)
        push!(pred_pers, T_test[i])
        push!(pred_clim, 0.0f0)
        push!(obs_H, T_test[i + H])
    end

    θ_samples = [θ.μ + exp.(θ.ρ) .* ComponentArray(randn(rng, Float32, length(θ.μ)), getaxes(θ.μ)) for m in 1:M]

    discrete_step_ude(u, p, t) = Float32[euler_step_bude(u[1], u[2], p, snn_f, snn_g, T_scale, h_scale; Δt=Δt)...]

    u0_dummy  = Float32[0.0, 0.0]
    prob_ude  = DiscreteProblem(discrete_step_ude, u0_dummy, (0, H), θ.μ)
    N_starts  = N_test - H

    function prob_func_ude(prob, idx, repeat)
        m = (idx - 1) % M + 1
        i = (idx - 1) ÷ M + 1
        remake(prob, u0=Float32[T_test[i], h_test[i]], p=θ_samples[m])
    end

    ensemble_prob_ude = EnsembleProblem(prob_ude; prob_func=prob_func_ude)
    sim_ude = solve(ensemble_prob_ude, FunctionMap(), EnsembleThreads(); trajectories=M*N_starts)

    pred_ude_mat = zeros(Float32, N_starts, M)
    for idx in 1:(M * N_starts)
        m = (idx - 1) % M + 1
        i = (idx - 1) ÷ M + 1
        pred_ude_mat[i, m] = sim_ude.u[idx].u[end][1]
    end
    pred_ude = mean(pred_ude_mat, dims=2)[:]

    rmse_ude_h[H],  r_ude_h[H]  = metrics(pred_ude,  obs_H)
    rmse_lin_h[H],  r_lin_h[H]  = metrics(pred_lin,  obs_H)
    rmse_ols_h[H],  r_ols_h[H]  = metrics(pred_ols,  obs_H)
    rmse_pers_h[H], r_pers_h[H] = metrics(pred_pers, obs_H)
    rmse_clim_h[H], r_clim_h[H] = metrics(pred_clim, obs_H)
end

@info "  H=6:  BUDE r=$(round(r_ude_h[6], digits=3)), Persistencia r=$(round(r_pers_h[6], digits=3)), OLS r=$(round(r_ols_h[6], digits=3))"
@info "  H=12: BUDE r=$(round(r_ude_h[12], digits=3)), Persistencia r=$(round(r_pers_h[12], digits=3)), OLS r=$(round(r_ols_h[12], digits=3))"

##############################################################################
# 5. Figuras
##############################################################################

const Figure = CairoMakie.Figure
const Axis   = CairoMakie.Axis

# Figura 1: Simulación larga en test
fig1 = Figure(size=(1100, 600), fontsize=13)
ax1a = Axis(fig1[1, 1], xlabel="Año", ylabel="Anomalía de SST (°C)",
            title="Test 2011–2021: simulación larga desde la condición inicial")
band!(ax1a, t_test, T_q10, T_q90; color=(:darkorange, 0.15), label="Intervalo de credibilidad del 90%")
lines!(ax1a, t_test, T_test;     color=:firebrick,   linewidth=1.8, label="Observado")
lines!(ax1a, t_test, T_ols_long; color=:gray,        linewidth=1.0, linestyle=:dash, label="OLS (r=$(round(r_ols_t, digits=2)))")
lines!(ax1a, t_test, T_lin_long; color=:dodgerblue,  linewidth=1.3, linestyle=:dash, label="BUDE Lin. (r=$(round(r_lin_t, digits=2)))")
lines!(ax1a, t_test, T_mean;     color=:darkorange,  linewidth=1.8, label="BUDE Media (r=$(round(r_ude_t, digits=2)))")
lines!(ax1a, t_test[1:n_min], T_tsit5[1:n_min]; color=:green, linewidth=1.3, linestyle=:dot, label="BUDE Tsit5 (r=$(round(r_tsit5_t, digits=2)))")
axislegend(ax1a, position=:lt, labelsize=11)

ax1b = Axis(fig1[2, 1], xlabel="Año", ylabel="Anomalía de WWV (10¹⁴ m³)", title="Profundidad de la termoclina (h)")
band!(ax1b, t_test, h_q10, h_q90; color=(:darkorange, 0.15))
lines!(ax1b, t_test, h_test;           color=:steelblue, linewidth=1.8, label="Observado")
lines!(ax1b, t_test, h_mean;           color=:darkorange, linewidth=1.8, label="BUDE Media")
lines!(ax1b, t_test[1:n_min], h_tsit5[1:n_min]; color=:green, linewidth=1.3, linestyle=:dot, label="BUDE Tsit5")
axislegend(ax1b, position=:lt, labelsize=11)

save("figures/04_test_rollout_bayesian.png", fig1; px_per_unit=2)
@info "Guardado figures/04_test_rollout_bayesian.png"

# Figura 2: Skill scores
fig2 = Figure(size=(1000, 500), fontsize=13)
Hs = 1:H_max

ax2a = Axis(fig2[1, 1], xlabel="Horizonte (meses)", ylabel="RMSE (°C)", title="Habilidad de pronóstico rodante — RMSE")
lines!(ax2a, Hs, rmse_clim_h; color=:black,      linewidth=1.5, linestyle=:dot,  label="Climatología")
lines!(ax2a, Hs, rmse_pers_h; color=:purple,     linewidth=1.5, linestyle=:dash, label="Persistencia")
lines!(ax2a, Hs, rmse_ols_h;  color=:gray,       linewidth=1.5, linestyle=:dash, label="OLS")
lines!(ax2a, Hs, rmse_lin_h;  color=:dodgerblue, linewidth=1.5, linestyle=:dash, label="BUDE Lin.")
lines!(ax2a, Hs, rmse_ude_h;  color=:darkorange, linewidth=2.0,                  label="BUDE Media")
axislegend(ax2a, position=:lt, labelsize=11)

ax2b = Axis(fig2[1, 2], xlabel="Horizonte (meses)", ylabel="Correlación", title="Habilidad de pronóstico rodante — Correlación (r)")
hlines!(ax2b, [0.0]; color=:gray, linewidth=0.8, linestyle=:dot)
lines!(ax2b, Hs, r_pers_h; color=:purple,     linewidth=1.5, linestyle=:dash, label="Persistencia")
lines!(ax2b, Hs, r_ols_h;  color=:gray,       linewidth=1.5, linestyle=:dash, label="OLS")
lines!(ax2b, Hs, r_lin_h;  color=:dodgerblue, linewidth=1.5, linestyle=:dash, label="BUDE Lin.")
lines!(ax2b, Hs, r_ude_h;  color=:darkorange, linewidth=2.0,                  label="BUDE Media")
axislegend(ax2b, position=:rt, labelsize=11)

save("figures/04_skill_score_bayesian.png", fig2; px_per_unit=2)
@info "Guardado figures/04_skill_score_bayesian.png"

# Figura 3: Espagueti a 6 meses
fig3 = Figure(size=(1100, 400), fontsize=13)
ax3 = Axis(fig3[1, 1], xlabel="Año", ylabel="Anomalía de SST (°C)",
           title="Pronósticos rodantes a 6 meses (BUDE) — incertidumbre de parámetros")
lines!(ax3, t_test, T_test; color=:firebrick, linewidth=2.0, label="Observado")

H6 = 6
spaghetti_M = 15
for i in 1:H6:(N_test - H6)
    t_seg = t_test[i:min(i + H6, N_test)]
    n_seg = length(t_seg)

    for m in 1:spaghetti_M
        ϵ_sample = ComponentArray(randn(rng, Float32, length(θ.μ)), getaxes(θ.μ))
        θ_sample = θ.μ + exp.(θ.ρ) .* ϵ_sample
        T_arr = zeros(Float32, n_seg)
        T_arr[1] = T_test[i]
        T_t, h_t = T_test[i], h_test[i]
        for k in 2:n_seg
            T_t, h_t = euler_step_bude(T_t, h_t, θ_sample, snn_f, snn_g, T_scale, h_scale; Δt=Δt)
            T_arr[k] = T_t
        end
        lines!(ax3, t_seg, T_arr; color=(:darkorange, 0.15), linewidth=1.0)
    end

    T_mean_seg = zeros(Float32, n_seg)
    T_mean_seg[1] = T_test[i]
    T_t, h_t = T_test[i], h_test[i]
    for k in 2:n_seg
        T_t, h_t = euler_step_bude(T_t, h_t, θ.μ, snn_f, snn_g, T_scale, h_scale; Δt=Δt)
        T_mean_seg[k] = T_t
    end
    lines!(ax3, t_seg, T_mean_seg; color=:darkorange, linewidth=1.8)
end

line_obs  = LineElement(color=:firebrick, linewidth=2.0)
line_mean = LineElement(color=:darkorange, linewidth=1.8)
line_memb = LineElement(color=(:darkorange, 0.3), linewidth=1.0)
Legend(fig3[1, 1], [line_obs, line_mean, line_memb], ["Observado", "Media BUDE", "Miembros del ensemble"],
       tellwidth=false, tellheight=false, halign=:right, valign=:top, margin=(10,10,10,10))

save("figures/04_rolling_6m_bayesian.png", fig3; px_per_unit=2)
@info "Guardado figures/04_rolling_6m_bayesian.png"

@info "===== Evaluación completada ====="