##############################################################################
# 05_sde_stochastic.jl  -  Ajuste de ruido de SDE y dinámica a largo plazo
#
# Metodología:
#   - Calcula residuos de predicción a un paso sobre los datos de entrenamiento.
#   - Ajusta modelos de ruido homocedástico y heterocedástico por MLE.
#   - Simula la SDE 500 años con Euler-Maruyama para estudiar la irregularidad
#     a largo plazo de ENSO.
#   - Detecta picos de El Niño y compara la distribución de intervalos
#     (observados vs. distintos modelos de ruido).
#
# Salidas:
#   data/sde_noise_params.jld2
#   figures/05_sde_analysis.png
#   figures/05_noise_surface.png
##############################################################################

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using DataFrames, CSV, Dates, JSON3, JLD2
using Lux, CairoMakie, ComponentArrays, Optim
using Random, Statistics, LinearAlgebra

cd(@__DIR__)
mkpath("data"); mkpath("figures")

isdefined(Main, :EnsoModels) || include("src/EnsoModels.jl")
using .EnsoModels

const rng = Xoshiro(42)
const Δt  = EnsoModels.Δt_default

##############################################################################
# 1. Cargar datos
##############################################################################

df       = CSV.read("data/enso_dataset.csv", DataFrame)
mask     = (year.(df.date) .>= 1980) .& (year.(df.date) .<= 2010)
df_train = df[mask, :]

T_obs = Float32.(df_train.T_anomaly)
h_obs = Float32.(df_train.h_anomaly)
dates = df_train.date
t_num = Float32.(year.(dates) .+ (month.(dates) .- 1) ./ 12)
N     = length(T_obs)

norm_params = JSON3.read(read("data/normalization.json", String))
T_scale     = Float32(norm_params["T_scale"])
h_scale     = Float32(norm_params["h_scale"])

T_all = Float32.(df.T_anomaly)
h_all = Float32.(df.h_anomaly)

p_base = JSON3.read(read("data/baseline_params.json", String))
a_ols  = Float32(p_base["a"])
b_ols  = Float32(p_base["b"])
c_ols  = Float32(p_base["c"])

##############################################################################
# 2. Reconstruir la deriva entrenada
##############################################################################

θ = load("data/ude_bayesian_params.jld2", "theta_bayes")

γ_T_joint = hasproperty(θ.μ, :γ_T) ? θ.μ.γ_T : nothing
γ_h_joint = hasproperty(θ.μ, :γ_h) ? θ.μ.γ_h : nothing

snn_f, snn_g, ps_f, ps_g = setup_bude_networks(rng)

drift_step(T_u, h_u) = euler_step_bude(T_u, h_u, θ.μ, snn_f, snn_g, T_scale, h_scale; Δt=Δt)

##############################################################################
# 3. Residuos de predicción a un paso
#    R = X_{t+Δt} - f(X_t)Δt  →  bajo una SDE de Itô, R ~ N(0, σ²Δt).
##############################################################################

R_T = zeros(Float32, N - 1)
R_h = zeros(Float32, N - 1)

for t in 1:(N - 1)
    T_pred, h_pred = drift_step(T_obs[t], h_obs[t])
    R_T[t] = T_obs[t+1] - T_pred
    R_h[t] = h_obs[t+1] - h_pred
end

@info "Residuos calculados."
@info "  Media R_T = $(mean(R_T)), Std R_T = $(std(R_T))"
@info "  Media R_h = $(mean(R_h)), Std R_h = $(std(R_h))"

##############################################################################
# 4. Ajuste de modelos de ruido por MLE
##############################################################################

# 4.1 Ruido homocedástico
σ_T_const = Float32(std(R_T) / sqrt(Δt))
σ_h_const = Float32(std(R_h) / sqrt(Δt))

@info "Ruido homocedástico: σ_T = $(round(σ_T_const, digits=4)), σ_h = $(round(σ_h_const, digits=4))"

# 4.2 Ruido heterocedástico: σ(T, h) = exp(γ[1] + γ[2]*T + γ[3]*h)
function nll_noise_T(γ)
    nll = 0.0
    for t in 1:(N - 1)
        sig = exp(γ[1] + γ[2]*T_obs[t] + γ[3]*h_obs[t])
        nll += 0.5 * log(2.0 * π * Δt * sig^2) + R_T[t]^2 / (2.0 * Δt * sig^2)
    end
    return nll
end

function nll_noise_h(γ)
    nll = 0.0
    for t in 1:(N - 1)
        sig = exp(γ[1] + γ[2]*T_obs[t+1] + γ[3]*h_obs[t])
        nll += 0.5 * log(2.0 * π * Δt * sig^2) + R_h[t]^2 / (2.0 * Δt * sig^2)
    end
    return nll
end

res_T = optimize(nll_noise_T, [log(σ_T_const), 0.0, 0.0], NelderMead())
res_h = optimize(nll_noise_h, [log(σ_h_const), 0.0, 0.0], NelderMead())

γ_T = Float32.(Optim.minimizer(res_T))
γ_h = Float32.(Optim.minimizer(res_h))

@info "Ruido heterocedástico (MLE secuencial):"
@info "  γ_T = [$(round(γ_T[1], digits=3)), $(round(γ_T[2], digits=3)), $(round(γ_T[3], digits=3))]"
@info "  γ_h = [$(round(γ_h[1], digits=3)), $(round(γ_h[2], digits=3)), $(round(γ_h[3], digits=3))]"

if !isnothing(γ_T_joint)
    @info "===== Comparación MLE vs ELBO ====="
    @info "  γ_T (MLE):  [$(round(γ_T[1], digits=3)), $(round(γ_T[2], digits=3)), $(round(γ_T[3], digits=3))]"
    @info "  γ_T (ELBO): [$(round(γ_T_joint[1], digits=3)), $(round(γ_T_joint[2], digits=3)), $(round(γ_T_joint[3], digits=3))]"
    @info "  γ_h (MLE):  [$(round(γ_h[1], digits=3)), $(round(γ_h[2], digits=3)), $(round(γ_h[3], digits=3))]"
    @info "  γ_h (ELBO): [$(round(γ_h_joint[1], digits=3)), $(round(γ_h_joint[2], digits=3)), $(round(γ_h_joint[3], digits=3))]"
end

jldsave("data/sde_noise_params.jld2"; σ_T_const=σ_T_const, σ_h_const=σ_h_const, γ_T=γ_T, γ_h=γ_h, γ_T_joint=γ_T_joint, γ_h_joint=γ_h_joint)
@info "Guardado data/sde_noise_params.jld2"

##############################################################################
# 5. Simulaciones SDE a largo plazo (500 años, Euler-Maruyama)
#    X_{t+Δt} = X_t + f(X_t)Δt + σ(X_t)ΔW,  ΔW ~ N(0, Δt)
##############################################################################

M_steps = 500 * 12

function simulate_sde(noise_type; seed=123)
    local_rng = Xoshiro(seed)
    T_sim = zeros(Float32, M_steps)
    h_sim = zeros(Float32, M_steps)
    T_t, h_t = 0.0f0, 0.0f0
    T_sim[1] = T_t
    h_sim[1] = h_t

    γ_T_sim = if noise_type == :none
        Float32[-Inf32, 0.0f0, 0.0f0]
    elseif noise_type == :homo
        Float32[log(σ_T_const), 0.0f0, 0.0f0]
    elseif noise_type == :joint && !isnothing(γ_T_joint)
        γ_T_joint
    else # :hetero
        γ_T
    end

    γ_h_sim = if noise_type == :none
        Float32[-Inf32, 0.0f0, 0.0f0]
    elseif noise_type == :homo
        Float32[log(σ_h_const), 0.0f0, 0.0f0]
    elseif noise_type == :joint && !isnothing(γ_h_joint)
        γ_h_joint
    else # :hetero
        γ_h
    end

    for k in 2:M_steps
        T_t, h_t = busde_step(
            local_rng, T_t, h_t, θ.μ, snn_f, snn_g, T_scale, h_scale,
            γ_T_sim, γ_h_sim; Δt=Δt
        )
        T_sim[k] = T_t
        h_sim[k] = h_t
    end
    return T_sim, h_sim
end

@info "Simulando escenarios SDE..."
T_det,    _         = simulate_sde(:none)
T_homo,   h_homo    = simulate_sde(:homo)
T_hetero, h_hetero  = simulate_sde(:hetero)
T_joint,  h_joint   = simulate_sde(:joint)

function simulate_persistence()
    local_rng = Xoshiro(42)
    T_sim  = zeros(Float32, M_steps)
    T_sim[1] = 0.0f0
    sig_pers = std(diff(T_obs)) / sqrt(Δt)
    for k in 2:M_steps
        T_sim[k] = T_sim[k-1] + sig_pers * randn(local_rng, Float32) * sqrt(Δt)
    end
    return T_sim
end
T_pers = simulate_persistence()

function simulate_ols()
    local_rng = Xoshiro(42)
    T_sim = zeros(Float32, M_steps)
    h_sim = zeros(Float32, M_steps)
    T_sim[1] = 0.0f0; h_sim[1] = 0.0f0

    R_T_ols = zeros(Float32, N-1)
    R_h_ols = zeros(Float32, N-1)
    for t in 1:N-1
        T_next, h_next = euler_step_ols(T_obs[t], h_obs[t], a_ols, b_ols, c_ols; Δt=Δt)
        R_T_ols[t] = T_obs[t+1] - T_next
        R_h_ols[t] = h_obs[t+1] - h_next
    end
    sig_T_ols = std(R_T_ols) / sqrt(Δt)
    sig_h_ols = std(R_h_ols) / sqrt(Δt)

    for k in 2:M_steps
        dW_T   = randn(local_rng, Float32) * sqrt(Δt)
        dW_h   = randn(local_rng, Float32) * sqrt(Δt)
        T_next = T_sim[k-1] + Δt * (a_ols * T_sim[k-1] + b_ols * h_sim[k-1]) + sig_T_ols * dW_T
        h_next = h_sim[k-1] + Δt * (-c_ols * T_next - a_ols * h_sim[k-1]) + sig_h_ols * dW_h
        T_sim[k] = T_next
        h_sim[k] = h_next
    end
    return T_sim
end
T_ols = simulate_ols()

##############################################################################
# 6. Detección de eventos y análisis de intervalos
#    Criterio ONI (NOAA): T > 0.5 °C durante al menos 5 meses consecutivos.
##############################################################################

function detect_events(T; threshold=0.5f0, min_duration=5)
    event_indices = Int[]
    n = length(T)
    i = 1
    while i <= n
        if T[i] > threshold
            j = i
            while j <= n && T[j] > threshold; j += 1; end
            if (j - i) >= min_duration
                push!(event_indices, i + argmax(T[i:(j-1)]) - 1)
            end
            i = j
        else
            i += 1
        end
    end
    return event_indices
end

obs_event_indices = detect_events(T_all)
obs_event_times   = obs_event_indices .* Δt
obs_intervals     = diff(obs_event_times)
obs_mean_int      = mean(obs_intervals)
obs_cv            = std(obs_intervals) / obs_mean_int

function analyze_scenario(T, name)
    evs = detect_events(T)
    if length(evs) < 3
        @info "  $name: demasiado pocos eventos para calcular intervalos."
        return Float32[]
    end
    ints  = diff(evs .* Δt)
    m_int = mean(ints)
    cv    = std(ints) / m_int
    @info "  $name: eventos=$(length(evs)), intervalo medio=$(round(m_int, digits=2)) años, CV=$(round(cv, digits=2))"
    return ints
end

@info "===== Análisis de intervalos entre eventos ====="
@info "  Observado: eventos=$(length(obs_event_indices)), intervalo medio=$(round(obs_mean_int, digits=2)) años, CV=$(round(obs_cv, digits=2))"
ints_det    = analyze_scenario(T_det,    "UDE determinista")
ints_homo   = analyze_scenario(T_homo,   "SDE homocedástica")
ints_hetero = analyze_scenario(T_hetero, "SDE heterocedástica (MLE)")
ints_joint  = analyze_scenario(T_joint,  "BUSDE conjunta")
ints_pers   = analyze_scenario(T_pers,   "Persistencia con ruido")
ints_ols    = analyze_scenario(T_ols,    "OLS con ruido homocedástico")

##############################################################################
# 7. Figuras
##############################################################################

const Figure = CairoMakie.Figure
const Axis   = CairoMakie.Axis

fig1 = Figure(size=(1400, 1400), fontsize=13)

ax1 = Axis(fig1[1, 1], xlabel="Intervalo (años)", ylabel="Densidad",
           title="Intervalos observados de El Niño (1980–2021) [CV = $(round(obs_cv, digits=2))]")
hist!(ax1, obs_intervals; bins=8, normalization=:pdf, color=(:firebrick, 0.6), strokecolor=:white, strokewidth=1)

ax2 = Axis(fig1[1, 2], xlabel="Intervalo (años)", ylabel="Densidad",
           title="SDE homocedástica [CV = $(length(ints_homo) > 0 ? round(std(ints_homo)/mean(ints_homo), digits=2) : NaN)]")
if length(ints_homo) > 0
    hist!(ax2, ints_homo; bins=15, normalization=:pdf, color=(:teal, 0.6), strokecolor=:white, strokewidth=1)
end

ax3 = Axis(fig1[2, 1], xlabel="Intervalo (años)", ylabel="Densidad",
           title="SDE heterocedástica (MLE) [CV = $(length(ints_hetero) > 0 ? round(std(ints_hetero)/mean(ints_hetero), digits=2) : NaN)]")
if length(ints_hetero) > 0
    hist!(ax3, ints_hetero; bins=15, normalization=:pdf, color=(:darkorange, 0.6), strokecolor=:white, strokewidth=1)
end

ax4 = Axis(fig1[2, 2], xlabel="Intervalo (años)", ylabel="Densidad",
           title="BUSDE conjunta [CV = $(length(ints_joint) > 0 ? round(std(ints_joint)/mean(ints_joint), digits=2) : NaN)]")
if length(ints_joint) > 0
    hist!(ax4, ints_joint; bins=15, normalization=:pdf, color=(:purple, 0.6), strokecolor=:white, strokewidth=1)
end

ax5 = Axis(fig1[3, 1], xlabel="Intervalo (años)", ylabel="Densidad",
           title="OLS con ruido homocedástico [CV = $(length(ints_ols) > 0 ? round(std(ints_ols)/mean(ints_ols), digits=2) : NaN)]")
if length(ints_ols) > 0
    hist!(ax5, ints_ols; bins=15, normalization=:pdf, color=(:dodgerblue, 0.6), strokecolor=:white, strokewidth=1)
end

ax6 = Axis(fig1[3, 2], xlabel="Anomalía de T (°C)", ylabel="Anomalía de h (10¹⁴ m³)",
           title="Espacio de fases SST-termoclina (segmentos SDE de 30 años)")
scatter!(ax6, T_all, h_all; color=(:gray, 0.4), markersize=3.5, label="Observado")
lines!(ax6, T_hetero[1:360], h_hetero[1:360]; color=:darkorange, linewidth=1.5, label="SDE hetero MLE")
lines!(ax6, T_joint[1:360],  h_joint[1:360];  color=:purple,     linewidth=1.5, label="BUSDE conjunta")
axislegend(ax6, position=:lt)

save("figures/05_sde_analysis.png", fig1; px_per_unit=2)
@info "Guardado figures/05_sde_analysis.png"

fig2 = Figure(size=(800, 600), fontsize=13)
ax_hm = Axis(fig2[1, 1], xlabel="Anomalía de T (°C)", ylabel="Anomalía de h (10¹⁴ m³)",
             title="Intensidad de ruido ajustada: σ_T(T, h) = exp(γ₀ + γ₁T + γ₂h)")

T_grid = range(-2.5f0, 2.5f0, length=50)
h_grid = range(-4.0f0, 4.0f0, length=50)
σ_grid = [exp(γ_T[1] + γ_T[2] * t + γ_T[3] * h) for t in T_grid, h in h_grid]

hm = heatmap!(ax_hm, collect(T_grid), collect(h_grid), σ_grid; colormap=:YlOrRd)
Colorbar(fig2[1, 2], hm, label="Desviación estándar del ruido de SST (σ_T)")
scatter!(ax_hm, T_obs, h_obs; color=(:black, 0.5), markersize=3.5)
axislegend(ax_hm, position=:lt)

save("figures/05_noise_surface.png", fig2; px_per_unit=2)
@info "Guardado figures/05_noise_surface.png"

@info "===== Análisis estocástico completado ====="
