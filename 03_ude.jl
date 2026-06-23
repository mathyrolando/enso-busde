##############################################################################
# 03_ude.jl  -  ENSO Bayesian UDE (BUDE)
#
# Metodología:
#   - Bayes-by-Backprop (BBB) para inferencia variacional.
#   - Estructura modular importando `EnsoModels.jl`.
#
# Salidas:
#   data/ude_bayesian_params.jld2
#   figures/03_ude_bayesian.png
#   figures/03_ude_nn_corrections.png
#   figures/03_ude_det_trajectory.png
##############################################################################

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using DataFrames, CSV, Dates, JSON3, JLD2
using Lux, Zygote, Optimisers
using CairoMakie, ComponentArrays
using Random, Statistics, LinearAlgebra

cd(@__DIR__)
mkpath("data")
mkpath("figures")

##############################################################################
# 1. Módulo y RNG
##############################################################################

isdefined(Main, :EnsoModels) || include("src/EnsoModels.jl")
using .EnsoModels

const rng = Xoshiro(42)

##############################################################################
# 2. Preparación de datos (ventana de entrenamiento 1980–2010)
##############################################################################

df       = CSV.read("data/enso_dataset.csv", DataFrame)
mask     = (year.(df.date) .>= 1980) .& (year.(df.date) .<= 2010)
df_train = df[mask, :]

T_obs = Float32.(df_train.T_anomaly)
h_obs = Float32.(df_train.h_anomaly)
N     = length(T_obs)

T_scale = Float32(std(T_obs))
h_scale = Float32(std(h_obs))

open("data/normalization.json", "w") do io
    JSON3.write(io, Dict("T_scale" => T_scale, "h_scale" => h_scale))
end

@info "Ventana de entrenamiento: $(df_train.date[1]) – $(df_train.date[end])  (N=$N)"
@info "T_scale=$(round(T_scale, digits=3))  h_scale=$(round(h_scale, digits=3))"

##############################################################################
# 3. Datos e hiperparámetros de entrenamiento
##############################################################################

const W = 36   # ventana de simulación de 3 años
const S = 6    # paso de 6 meses
starts  = collect(1:S:(N - W + 1))

train_data = EnsoTrainingData(
    T_obs, h_obs, starts, W,
    T_scale, h_scale, EnsoModels.Δt_default
)

##############################################################################
# 4. Inicialización del modelo y prior
##############################################################################

p_base = JSON3.read("data/baseline_params.json")
a0 = clamp(Float32(p_base["a"]), -0.40f0, -0.05f0)
b0 = max(Float32(p_base["b"]), 0.05f0)
c0 = max(Float32(p_base["c"]), 0.05f0)

snn_f, snn_g, ps_f, ps_g = setup_bude_networks(rng)

μ0 = ComponentArray(
    a     = a0,
    log_b = log(b0),
    log_c = log(c0),
    f     = ComponentArray(ps_f),
    g     = ComponentArray(ps_g),
    γ_T   = Float32[-0.4f0, 0f0, 0f0],
    γ_h   = Float32[0.3f0, 0f0, 0f0]
)

ρ0 = similar(μ0)
ρ0 .= -3.9f0

θ_bayes = ComponentArray(μ = μ0, ρ = ρ0)

σ_prior_vec = copy(μ0)
σ_prior_vec.a = 2.0f0
σ_prior_vec.log_b = 2.0f0
σ_prior_vec.log_c = 2.0f0
σ_prior_vec.f .= 0.5f0
σ_prior_vec.g .= 0.5f0
σ_prior_vec.γ_T .= Float32[1.0f0, 0.5f0, 0.5f0]
σ_prior_vec.γ_h .= Float32[1.0f0, 0.5f0, 0.5f0]
const σ_prior_flat = Float32.(vec(σ_prior_vec))

##############################################################################
# 5. Entrenamiento bayesiano (1000 iteraciones)
##############################################################################

loss_history = Float32[]
opt = Optimisers.setup(Optimisers.Adam(1f-3), θ_bayes)

@info "Entrenando la UDE bayesiana (1000 iteraciones)..."
for iter in 1:1000
    global θ_bayes, opt

    if iter == 601
        opt = Optimisers.setup(Optimisers.Adam(3f-4), θ_bayes)
        @info "  Tasa de aprendizaje → 3e-4 en iteración 601"
    end

    ϵ = ComponentArray(randn(rng, Float32, length(μ0)), getaxes(μ0))

    ℓ, ∇ = Zygote.withgradient(p -> elbo_joint_loss(p, ϵ, train_data, snn_f, snn_g, σ_prior_flat), θ_bayes)
    push!(loss_history, ℓ)

    if isfinite(ℓ) && !isnothing(∇[1])
        opt, θ_bayes = Optimisers.update!(opt, θ_bayes, ∇[1])
    end

    θ_bayes.μ.a = clamp(θ_bayes.μ.a, -0.40f0, -0.05f0)

    if iter % 100 == 0 || iter == 1
        a_mean  = round(θ_bayes.μ.a, digits=3)
        b_mean  = round(exp(θ_bayes.μ.log_b), digits=3)
        γT_mean = round.(θ_bayes.μ.γ_T, digits=3)
        @info "  iter $iter/1000  ELBO=$(round(ℓ, sigdigits=4))  a=$a_mean  b=$b_mean  γ_T=$γT_mean"
    end
end

@info "Entrenamiento completado."
jldsave("data/ude_bayesian_params.jld2"; theta_bayes=θ_bayes)

##############################################################################
# 6. Evaluación en el conjunto de entrenamiento
##############################################################################

function ms_reconstruction(p, data::EnsoTrainingData, snn_f, snn_g)
    n = length(data.T_obs)
    T_pred = fill(NaN32, n)
    T_pred[1] = data.T_obs[1]
    for s in 1:S:(n - data.W + 1)
        T_t, h_t = data.T_obs[s], data.h_obs[s]
        for k in 1:(data.W - 1)
            T_t, h_t = euler_step_bude(
                T_t, h_t, p, snn_f, snn_g,
                data.T_scale, data.h_scale, Δt=data.Δt
            )
            idx = s + k
            if idx <= n && isnan(T_pred[idx])
                T_pred[idx] = T_t
            end
        end
    end
    return T_pred
end

function onestep_predictions(p, data::EnsoTrainingData, snn_f, snn_g)
    n = length(data.T_obs)
    T_1step = zeros(Float32, n)
    T_1step[1] = data.T_obs[1]
    for t in 1:(n-1)
        T_next, _ = euler_step_bude(
            data.T_obs[t], data.h_obs[t], p, snn_f, snn_g,
            data.T_scale, data.h_scale, Δt=data.Δt
        )
        T_1step[t+1] = T_next
    end
    return T_1step
end

T_ms = ms_reconstruction(θ_bayes.μ, train_data, snn_f, snn_g)
T_1s = onestep_predictions(θ_bayes.μ, train_data, snn_f, snn_g)

valid   = .!isnan.(T_ms)
rmse_ms = sqrt(mean((T_ms[valid] .- T_obs[valid]).^2))
r_ms    = cor(T_ms[valid], T_obs[valid])
rmse_1s = sqrt(mean((T_1s[2:end] .- T_obs[2:end]).^2))
r_1s    = cor(T_1s[2:end], T_obs[2:end])

@info "  1 paso: RMSE=$(round(rmse_1s, digits=3)), r=$(round(r_1s, digits=3))"
@info "  MS ($W m): RMSE=$(round(rmse_ms, digits=3)), r=$(round(r_ms, digits=3))"

##############################################################################
# 7. Graficado del entrenamiento
##############################################################################

t_num = Float32.(year.(df_train.date) .+ (month.(df_train.date) .- 1) ./ 12)

const Figure = CairoMakie.Figure
const Axis   = CairoMakie.Axis

fig = Figure(size=(1400, 1050), fontsize=13)

ax1 = Axis(fig[1, 1:2], title="Predicción a 1 paso [r = $(round(r_1s, digits=3))]")
lines!(ax1, t_num, T_obs; color=:firebrick, linewidth=1.5, label="Observado")
lines!(ax1, t_num, T_1s; color=:darkorange, linewidth=1.8, label="BUDE 1 paso")
axislegend(ax1, position=:lt)

ax2 = Axis(fig[2, 1:2], title="Reconstrucción MS ($W meses) [r = $(round(r_ms, digits=3))]")
lines!(ax2, t_num, T_obs; color=:firebrick, linewidth=1.5)
for s in 1:S:(N - W + 1)
    T_t, h_t = T_obs[s], h_obs[s]
    T_win = Float32[T_obs[s]]
    for k in 1:(W - 1)
        T_t, h_t = euler_step_bude(
            T_t, h_t, θ_bayes.μ, snn_f, snn_g,
            T_scale, h_scale, Δt=EnsoModels.Δt_default
        )
        push!(T_win, T_t)
    end
    lines!(ax2, t_num[s:min(s+W-1, N)], T_win; color=(:darkorange, 0.4))
end

ax3 = Axis(fig[3, 1], title="Espacio de fases")
scatter!(ax3, T_obs, h_obs; color=:firebrick, markersize=3, alpha=0.5)
scatter!(ax3, T_1s, h_obs; color=:darkorange, markersize=3, alpha=0.5)

ax4 = Axis(fig[3, 2], title="Pérdida ELBO conjunta")
lines!(ax4, 1:length(loss_history), loss_history; color=:darkorange, linewidth=1.5)

save("figures/03_ude_bayesian.png", fig; px_per_unit=2)
@info "Guardado figures/03_ude_bayesian.png"

##############################################################################
# 8. Correcciones aprendidas por la red neuronal
##############################################################################

@info "Graficando las correcciones aprendidas por la red neuronal..."
fig_nn = Figure(size=(1200, 500), fontsize=13)
ax_f = Axis(fig_nn[1, 1], xlabel="Anomalía de T (°C)", ylabel="Anomalía de h (10¹⁴ m³)",
            title="Corrección de SST aprendida: f(T, h)")
ax_g = Axis(fig_nn[1, 3], xlabel="Anomalía de T (°C)", ylabel="Anomalía de h (10¹⁴ m³)",
            title="Corrección de la termoclina aprendida: g(T, h)")

T_grid = range(-2.5f0, 2.5f0, length=50)
h_grid = range(-4.0f0, 4.0f0, length=50)

f_grid = [only(snn_f(features(t, h, T_scale, h_scale), θ_bayes.μ.f)) for t in T_grid, h in h_grid]
g_grid = [only(snn_g(features(t, h, T_scale, h_scale), θ_bayes.μ.g)) for t in T_grid, h in h_grid]

clim_f = max(abs(minimum(f_grid)), abs(maximum(f_grid)))
clim_g = max(abs(minimum(g_grid)), abs(maximum(g_grid)))

hm_f = heatmap!(ax_f, collect(T_grid), collect(h_grid), f_grid; colormap=:coolwarm, colorrange=(-clim_f, clim_f))
Colorbar(fig_nn[1, 2], hm_f, label="f(T, h) [°C/año]")
scatter!(ax_f, T_obs, h_obs; color=(:black, 0.4), markersize=3)

hm_g = heatmap!(ax_g, collect(T_grid), collect(h_grid), g_grid; colormap=:coolwarm, colorrange=(-clim_g, clim_g))
Colorbar(fig_nn[1, 4], hm_g, label="g(T, h) [10¹⁴ m³/año]")
scatter!(ax_g, T_obs, h_obs; color=(:black, 0.4), markersize=3)

save("figures/03_ude_nn_corrections.png", fig_nn; px_per_unit=2)
@info "Guardado figures/03_ude_nn_corrections.png"

# Reporte de parámetros de la posterior
open("data/bude_posterior_report.txt", "w") do io
    println(io, "=== Reporte de Parámetros de la Posterior ===")
    println(io, "a: media = ", round(θ_bayes.μ.a, digits=4), ", std = ", round(exp(θ_bayes.ρ.a), digits=4))
    println(io, "b: media = ", round(exp(θ_bayes.μ.log_b), digits=4), ", std = ", round(exp(θ_bayes.ρ.log_b), digits=4))
    println(io, "c: media = ", round(exp(θ_bayes.μ.log_c), digits=4), ", std = ", round(exp(θ_bayes.ρ.log_c), digits=4))
    b_val = exp(θ_bayes.μ.log_b)
    c_val = exp(θ_bayes.μ.log_c)
    println(io, "Periodo físico: ", round(2*π / sqrt(b_val * c_val), digits=2), " años")
end
@info "Guardado data/bude_posterior_report.txt"

##############################################################################
# 9. Trayectoria determinista libre a largo plazo (50 años)
##############################################################################

@info "Simulando trayectoria determinista libre a 50 años..."
M_steps = 50 * 12
T_det = zeros(Float32, M_steps)
h_det = zeros(Float32, M_steps)
T_det[1] = T_obs[1]
h_det[1] = h_obs[1]

a_p = θ_bayes.μ.a
b_p = exp(θ_bayes.μ.log_b)
c_p = exp(θ_bayes.μ.log_c)

for k in 2:M_steps
    x     = features(T_det[k-1], h_det[k-1], T_scale, h_scale)
    f_out = only(snn_f(x, θ_bayes.μ.f))
    g_out = only(snn_g(x, θ_bayes.μ.g))
    T_next    = T_det[k-1] + EnsoModels.Δt_default * (a_p * T_det[k-1] + b_p * h_det[k-1] + f_out)
    h_next    = h_det[k-1] + EnsoModels.Δt_default * (-c_p * T_next - a_p * h_det[k-1] + g_out)
    T_det[k]  = T_next
    h_det[k]  = h_next
end

fig_det    = Figure(size=(1100, 420), fontsize=13)
t_det_years = collect(1:M_steps) .* Float64(EnsoModels.Δt_default)

ax_det = Axis(fig_det[1, 1],
    xlabel = "Tiempo (años)",
    ylabel = "Anomalía de T (°C)",
    title  = "Trayectoria determinista libre de la UDE - 50 años")
lines!(ax_det, t_det_years, T_det; color=:steelblue, linewidth=1.8)
hlines!(ax_det,  0.5f0; color=(:firebrick, 0.7), linewidth=1.2, linestyle=:dash, label="Umbral El Niño (+0.5 °C)")
hlines!(ax_det, -0.5f0; color=(:royalblue, 0.7), linewidth=1.2, linestyle=:dash, label="Umbral La Niña (−0.5 °C)")
hlines!(ax_det,  0.0f0; color=(:black, 0.3),     linewidth=0.8, linestyle=:dot)
axislegend(ax_det, position=:rt)

ax_ph = Axis(fig_det[1, 2],
    xlabel = "Anomalía de T (°C)",
    ylabel = "Anomalía de h (10¹⁴ m³)",
    title  = "Espacio de fases determinista (50 años)")
lines!(ax_ph, T_det, h_det; color=:steelblue, linewidth=1.2)
scatter!(ax_ph, [T_det[1]], [h_det[1]]; color=:green, markersize=10, label="Inicio")
axislegend(ax_ph, position=:lt)

save("figures/03_ude_det_trajectory.png", fig_det; px_per_unit=2)
@info "Guardado figures/03_ude_det_trajectory.png"