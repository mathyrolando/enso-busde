module EnsoModels

using Lux
using ComponentArrays
using Random
using Statistics
using SpecialFunctions: erf
using DifferentialEquations: ODEProblem, solve, Tsit5
using StochasticDiffEq: SDEProblem, solve, EM

##############################################################################
# Exportaciones
##############################################################################
export Δt_default
export EnsoTrainingData
export features, setup_bude_networks
export euler_step_ols, euler_step_bude, busde_step
export crps_ensemble, crps_climatology
export elbo_loss, deterministic_loss
export solve_bude_ode, solve_busde_sde, elbo_joint_loss

##############################################################################
# Constantes
##############################################################################
const Δt_default = 1.0f0 / 12.0f0   # un mes en años

##############################################################################
# Estructuras de datos
##############################################################################

"""
Encapsula los datos de entrenamiento y los hiperparámetros de ventana
para evitar variables globales en el cálculo del gradiente.
"""
struct EnsoTrainingData
    T_obs::Vector{Float32}
    h_obs::Vector{Float32}
    starts::Vector{Int}
    W::Int
    T_scale::Float32
    h_scale::Float32
    Δt::Float32
end

##############################################################################
# Features y arquitectura de red
##############################################################################

function features(T_u::Float32, h_u::Float32, T_scale::Float32, h_scale::Float32, version::String="A")
    T̃ = T_u / T_scale
    h̃ = h_u / h_scale
    if version == "A"
        return Float32[T̃, h̃, T̃^2, h̃^2, T̃ * h̃]
    elseif version == "B"
        return Float32[T̃, h̃, T̃^2, h̃^2]
    elseif version == "C"
        return Float32[T̃, h̃]
    else
        error("Unknown features version: $version")
    end
end

function setup_bude_networks(rng::AbstractRNG, n_features::Int=5)
    small_init(r, out, in) = randn(r, Float32, out, in) .* 1f-2

    nn_f = Lux.Chain(
        Lux.Dense(n_features => 16, tanh; use_bias=false),
        Lux.Dense(16 => 1; init_weight=small_init, use_bias=false)
    )
    nn_g = Lux.Chain(
        Lux.Dense(n_features => 16, tanh; use_bias=false),
        Lux.Dense(16 => 1; init_weight=small_init, use_bias=false)
    )

    ps_f, st_f = Lux.setup(rng, nn_f)
    ps_g, st_g = Lux.setup(rng, nn_g)

    snn_f = Lux.StatefulLuxLayer{true}(nn_f, ps_f, st_f)
    snn_g = Lux.StatefulLuxLayer{true}(nn_g, ps_g, st_g)

    return snn_f, snn_g, ps_f, ps_g
end

##############################################################################
# Integradores
##############################################################################

# Oscilador de recarga lineal (Euler simpléctico).
# Usado tanto para el baseline OLS como para el componente lineal del BUDE;
# la distinción la hace el llamador al pasar distintos parámetros.
function euler_step_ols(T_u::Float32, h_u::Float32, a::Float32, b::Float32, c::Float32; Δt=Δt_default)
    T_next = T_u + Δt * (a * T_u + b * h_u)
    h_next = h_u + Δt * (-c * T_next - a * h_u)
    return T_next, h_next
end

# UDE determinista (BUDE)
function euler_step_bude(T_u::Float32, h_u::Float32, p, snn_f, snn_g, T_scale::Float32, h_scale::Float32; Δt=Δt_default, version::String="A")
    a_p = p.a
    b_p = exp(p.log_b)
    c_p = exp(p.log_c)

    x     = features(T_u, h_u, T_scale, h_scale, version)
    f_out = only(snn_f(x, p.f))
    g_out = only(snn_g(x, p.g))

    T_next = T_u + Δt * (a_p * T_u + b_p * h_u + f_out)
    h_next = h_u + Δt * (-c_p * T_next - a_p * h_u + g_out)

    return T_next, h_next
end

# SDE estocástica (BUSDE) con Euler-Maruyama simpléctico
function busde_step(rng::AbstractRNG, T_u::Float32, h_u::Float32, p, snn_f, snn_g, T_scale::Float32, h_scale::Float32, γ_T=nothing, γ_h=nothing; Δt=Δt_default, version::String="A")
    a_p = p.a
    b_p = exp(p.log_b)
    c_p = exp(p.log_c)

    γ_T_val = isnothing(γ_T) ? (hasproperty(p, :γ_T) ? p.γ_T : Float32[-Inf32, 0.0f0, 0.0f0]) : γ_T
    γ_h_val = isnothing(γ_h) ? (hasproperty(p, :γ_h) ? p.γ_h : Float32[-Inf32, 0.0f0, 0.0f0]) : γ_h

    x     = features(T_u, h_u, T_scale, h_scale, version)
    f_out = only(snn_f(x, p.f))
    g_out = only(snn_g(x, p.g))

    sig_T  = exp(γ_T_val[1] + γ_T_val[2] * T_u + γ_T_val[3] * h_u)
    dW_T   = randn(rng, Float32) * sqrt(Δt)
    T_next = T_u + Δt * (a_p * T_u + b_p * h_u + f_out) + sig_T * dW_T

    # Simpléctico: σ_h usa T_next para consistencia con la actualización de h
    sig_h  = exp(γ_h_val[1] + γ_h_val[2] * T_next + γ_h_val[3] * h_u)
    dW_h   = randn(rng, Float32) * sqrt(Δt)
    h_next = h_u + Δt * (-c_p * T_next - a_p * h_u + g_out) + sig_h * dW_h

    return T_next, h_next
end

##############################################################################
# Funciones de pérdida
##############################################################################

function elbo_loss(p, ϵ, data::EnsoTrainingData, snn_f_layer, snn_g_layer, prior_flat; version::String="A")
    θ = p.μ + exp.(p.ρ) .* ϵ

    total = 0f0
    valid = 0

    for s in data.starts
        T_t = data.T_obs[s]
        h_t = data.h_obs[s]
        win_loss = 0f0

        for k in 1:(data.W - 1)
            T_t, h_t = euler_step_bude(
                T_t, h_t, θ, snn_f_layer, snn_g_layer,
                data.T_scale, data.h_scale; Δt=data.Δt, version=version
            )
            eT = (T_t - data.T_obs[s + k]) / data.T_scale
            eh = (h_t - data.h_obs[s + k]) / data.h_scale
            win_loss += eT^2 + eh^2
        end
        total += win_loss / (data.W - 1)
        valid += 1
    end

    loss_lik = valid == 0 ? 1f4 : total / valid

    μ_flat = vec(p.μ)
    ρ_flat = vec(p.ρ)
    σ_flat = exp.(ρ_flat)
    kl     = log.(prior_flat) .- ρ_flat .+ (σ_flat.^2 .+ μ_flat.^2) ./ (2.0f0 .* prior_flat.^2) .- 0.5f0
    loss_kl = sum(kl) * (1.0f0 / Float32(length(data.T_obs)))

    return loss_lik + loss_kl
end

function deterministic_loss(p, data::EnsoTrainingData, snn_f_layer, snn_g_layer; version::String="A")
    total = 0f0
    valid = 0
    for s in data.starts
        T_t = data.T_obs[s]
        h_t = data.h_obs[s]
        win_loss = 0f0
        for k in 1:(data.W - 1)
            T_t, h_t = euler_step_bude(
                T_t, h_t, p, snn_f_layer, snn_g_layer,
                data.T_scale, data.h_scale; Δt=data.Δt, version=version
            )
            eT = (T_t - data.T_obs[s + k]) / data.T_scale
            eh = (h_t - data.h_obs[s + k]) / data.h_scale
            win_loss += eT^2 + eh^2
        end
        total += win_loss / (data.W - 1)
        valid += 1
    end
    return valid == 0 ? 1f4 : total / valid
end

##############################################################################
# Solvers continuos (SciML)
##############################################################################

function solve_bude_ode(T0, h0, p, snn_f, snn_g, T_scale::Float32, h_scale::Float32, tspan=(0f0, 3f0); solver=Tsit5(), Δt=Δt_default, version::String="A")
    function bude_ode!(du, u, p_ode, t)
        T_u, h_u = u[1], u[2]
        a_p = p_ode.a
        b_p = exp(p_ode.log_b)
        c_p = exp(p_ode.log_c)
        x     = features(T_u, h_u, T_scale, h_scale, version)
        f_out = only(snn_f(x, p_ode.f))
        g_out = only(snn_g(x, p_ode.g))
        du[1] = a_p * T_u + b_p * h_u + f_out
        du[2] = -c_p * T_u - a_p * h_u + g_out
    end

    u0   = Float32[T0, h0]
    prob = ODEProblem(bude_ode!, u0, tspan, p)
    sol  = solve(prob, solver; saveat=Δt)
    return sol
end

function solve_busde_sde(T0, h0, p, snn_f, snn_g, T_scale::Float32, h_scale::Float32, tspan, local_rng; solver=EM(), Δt=Δt_default, version::String="A")
    function drift!(du, u, p_sde, t)
        T_u, h_u = u[1], u[2]
        a_p = p_sde.a
        b_p = exp(p_sde.log_b)
        c_p = exp(p_sde.log_c)
        x     = features(T_u, h_u, T_scale, h_scale, version)
        f_out = only(snn_f(x, p_sde.f))
        g_out = only(snn_g(x, p_sde.g))
        du[1] = a_p * T_u + b_p * h_u + f_out
        du[2] = -c_p * T_u - a_p * h_u + g_out
    end

    function noise!(du, u, p_sde, t)
        T_u, h_u = u[1], u[2]
        γ_T_val = hasproperty(p_sde, :γ_T) ? p_sde.γ_T : Float32[-Inf32, 0.0f0, 0.0f0]
        γ_h_val = hasproperty(p_sde, :γ_h) ? p_sde.γ_h : Float32[-Inf32, 0.0f0, 0.0f0]
        du[1] = exp(γ_T_val[1] + γ_T_val[2] * T_u + γ_T_val[3] * h_u)
        du[2] = exp(γ_h_val[1] + γ_h_val[2] * T_u + γ_h_val[3] * h_u)
    end

    u0   = Float32[T0, h0]
    prob = SDEProblem(drift!, noise!, u0, tspan, p)
    sol  = solve(prob, solver; dt=Δt, saveat=Δt, rng=local_rng)
    return sol
end

# Pérdida ELBO conjunta: optimiza deriva y difusión simultáneamente
function elbo_joint_loss(p, ϵ, data::EnsoTrainingData, snn_f_layer, snn_g_layer, prior_flat; version::String="A")
    θ = p.μ + exp.(p.ρ) .* ϵ

    # 1. Pérdida multi-step determinista
    total_ms = 0f0
    valid_ms = 0

    for s in data.starts
        T_t = data.T_obs[s]
        h_t = data.h_obs[s]
        win_loss = 0f0

        for k in 1:(data.W - 1)
            T_t, h_t = euler_step_bude(
                T_t, h_t, θ, snn_f_layer, snn_g_layer,
                data.T_scale, data.h_scale; Δt=data.Δt, version=version
            )
            eT = (T_t - data.T_obs[s + k]) / data.T_scale
            eh = (h_t - data.h_obs[s + k]) / data.h_scale
            win_loss += eT^2 + eh^2
        end
        total_ms += win_loss / (data.W - 1)
        valid_ms += 1
    end
    loss_ms = valid_ms == 0 ? 1f4 : total_ms / valid_ms

    # 2. Log-verosimilitud negativa de las transiciones a un paso de la SDE
    γ_T = θ.γ_T
    γ_h = θ.γ_h
    a_p = θ.a
    b_p = exp(θ.log_b)
    c_p = exp(θ.log_c)

    n = length(data.T_obs)

    T_curr_all = @view data.T_obs[1:(n-1)]
    h_curr_all = @view data.h_obs[1:(n-1)]
    T_next_all = @view data.T_obs[2:n]
    h_next_all = @view data.h_obs[2:n]

    T̃_curr = T_curr_all ./ data.T_scale
    h̃_curr = h_curr_all ./ data.h_scale
    T̃_next = T_next_all ./ data.T_scale

    T̃_curr_sq = T̃_curr .* T̃_curr
    h̃_curr_sq = h̃_curr .* h̃_curr
    T̃_next_sq = T̃_next .* T̃_next

    if version == "A"
        X_curr = [T̃_curr'; h̃_curr'; T̃_curr_sq'; h̃_curr_sq'; (T̃_curr .* h̃_curr)']
        X_next = [T̃_next'; h̃_curr'; T̃_next_sq'; h̃_curr_sq'; (T̃_next .* h̃_curr)']
    elseif version == "B"
        X_curr = [T̃_curr'; h̃_curr'; T̃_curr_sq'; h̃_curr_sq']
        X_next = [T̃_next'; h̃_curr'; T̃_next_sq'; h̃_curr_sq']
    else
        X_curr = [T̃_curr'; h̃_curr']
        X_next = [T̃_next'; h̃_curr']
    end

    f_out_all = vec(snn_f_layer(X_curr, θ.f))
    g_out_all = vec(snn_g_layer(X_next, θ.g))

    T_pred = T_curr_all .+ data.Δt .* (a_p .* T_curr_all .+ b_p .* h_curr_all .+ f_out_all)
    sig_T  = exp.(γ_T[1] .+ γ_T[2] .* T_curr_all .+ γ_T[3] .* h_curr_all)
    res_T  = T_next_all .- T_pred
    sig_T_stab    = sig_T .+ 1f-4
    sig_T_stab_sq = sig_T_stab .* sig_T_stab
    nll_T = 0.5f0 .* log.(2.0f0 .* Float32(π) .* data.Δt .* sig_T_stab_sq) .+ (res_T .* res_T) ./ (2.0f0 .* data.Δt .* sig_T_stab_sq)

    h_pred = h_curr_all .+ data.Δt .* (-c_p .* T_next_all .- a_p .* h_curr_all .+ g_out_all)
    sig_h  = exp.(γ_h[1] .+ γ_h[2] .* T_next_all .+ γ_h[3] .* h_curr_all)
    res_h  = h_next_all .- h_pred
    sig_h_stab    = sig_h .+ 1f-4
    sig_h_stab_sq = sig_h_stab .* sig_h_stab
    nll_h = 0.5f0 .* log.(2.0f0 .* Float32(π) .* data.Δt .* sig_h_stab_sq) .+ (res_h .* res_h) ./ (2.0f0 .* data.Δt .* sig_h_stab_sq)

    loss_sde = (sum(nll_T) + sum(nll_h)) / (n - 1)

    # 3. Término KL (Bayes-by-Backprop)
    μ_flat  = vec(p.μ)
    ρ_flat  = vec(p.ρ)
    σ_flat  = exp.(ρ_flat)
    kl      = log.(prior_flat) .- ρ_flat .+ (σ_flat.^2 .+ μ_flat.^2) ./ (2.0f0 .* prior_flat.^2) .- 0.5f0
    loss_kl = sum(kl) * (1.0f0 / Float32(n))

    return loss_ms + loss_sde + loss_kl
end

##############################################################################
# Métricas de verificación
##############################################################################

function crps_ensemble(X::AbstractVector, y::Real)
    M = length(X)
    mae = mean(abs.(X .- y))
    X_sorted = sort(X)
    spread = 0.0
    for i in 1:M
        spread += (2.0 * i - M - 1.0) * X_sorted[i]
    end
    spread = spread / (M^2)
    return mae - spread
end

function crps_climatology(y::Real, T_train::AbstractVector)
    μ_cl  = mean(T_train)
    σ_cl  = std(T_train)
    z     = (y - μ_cl) / σ_cl
    cdf_z = 0.5 * (1.0 + erf(z / sqrt(2.0)))
    pdf_z = exp(-0.5 * z^2) / sqrt(2.0 * π)
    return σ_cl * (z * (2.0 * cdf_z - 1.0) + 2.0 * pdf_z - 1.0 / sqrt(π))
end

end