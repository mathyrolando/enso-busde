##############################################################################
# 07_bude_vs_busde.jl  -  Comparación probabilística: BUDE vs BUSDE
#
# Metodología:
#   - BUDE (UDE Bayesiana): solo incertidumbre epistémica - ensemble de M
#     muestras de la posterior q(θ) = N(μ, exp(2ρ)), corridas deterministas.
#   - BUSDE (SDE Universal Bayesiana): epistémica + aleatoria - mismo
#     muestreo de parámetros pero con ruido SDE heterocedástico ajustado
#     por MLE en 05_sde_stochastic.jl.
#
# Hallazgo clave:
#   BUDE está severamente subdispersado porque ignora el ruido aleatorio.
#   BUSDE está mejor calibrado en todos los horizontes:
#     1. CRPS más bajo.
#     2. Cobertura de PI al 90% más cercana al nominal.
#     3. Diagrama de confiabilidad más diagonal a H = 6 meses.
#     4. Bandas de ensemble visiblemente más anchas en el gráfico de abanico.
#
# Salidas:
#   figures/07_bude_vs_busde.png
##############################################################################

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using DataFrames, CSV, Dates, JLD2, JSON3
using Lux, CairoMakie, ComponentArrays
using Random, Statistics, LinearAlgebra, SpecialFunctions, Printf

cd(@__DIR__)
mkpath("figures")

const rng = Xoshiro(42)
const Δt  = Float32(1.0 / 12.0)

##############################################################################
# 1. Cargar datos
##############################################################################

df = CSV.read("data/enso_dataset.csv", DataFrame)

norm_params = JSON3.read(read("data/normalization.json", String))
T_scale     = Float32(norm_params["T_scale"])
h_scale     = Float32(norm_params["h_scale"])

mask_train = (year.(df.date) .>= 1980) .& (year.(df.date) .<= 2010)
T_train    = Float32.(df[mask_train, :].T_anomaly)

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

θ_bayes = load("data/ude_bayesian_params.jld2", "theta_bayes")
sde_par = load("data/sde_noise_params.jld2")
γ_T = Float32.(sde_par["γ_T"])
γ_h = Float32.(sde_par["γ_h"])

isdefined(Main, :EnsoModels) || include("src/EnsoModels.jl")
using .EnsoModels

snn_f, snn_g, ps_f, ps_g = setup_bude_networks(rng)

bude_step(T_u, h_u, p) = EnsoModels.euler_step_bude(T_u, h_u, p, snn_f, snn_g, T_scale, h_scale; Δt=Δt)

##############################################################################
# 3. Pronósticos rodantes H = 1..12
##############################################################################

H_max = 12
M     = 500

crps_bude   = zeros(H_max); crps_busde  = zeros(H_max); crps_clima = zeros(H_max)
cov_bude    = zeros(H_max); cov_busde   = zeros(H_max)
brier_bude  = zeros(H_max); brier_busde = zeros(H_max)

probs_bude_6m  = Float32[]
probs_busde_6m = Float32[]
obs_events_6m  = Float32[]

@info "Calculando pronósticos rodantes BUDE y BUSDE (M=$M, H=1..$H_max)..."
for H in 1:H_max
    @info "  H=$H..."
    crps_b_list  = Float64[]; crps_s_list  = Float64[]; crps_c_list = Float64[]
    brier_b_list = Float64[]; brier_s_list = Float64[]
    cov_b = 0; cov_s = 0
    n_valid = N_test - H

    for i in 1:n_valid
        T0, h0 = T_test[i], h_test[i]
        y_true = T_test[i + H]

        pred_bude_arr  = zeros(Float32, M)
        pred_busde_arr = zeros(Float32, M)

        for m in 1:M
            ϵ   = ComponentArray(randn(Float32, length(θ_bayes.μ)), getaxes(θ_bayes.μ))
            θ_s = θ_bayes.μ + exp.(θ_bayes.ρ) .* ϵ

            T_b, h_b = T0, h0
            for _ in 1:H
                T_b, h_b = bude_step(T_b, h_b, θ_s)
            end
            pred_bude_arr[m] = T_b

            T_s, h_s = T0, h0
            for _ in 1:H
                T_s, h_s = EnsoModels.busde_step(rng, T_s, h_s, θ_s, snn_f, snn_g, T_scale, h_scale, γ_T, γ_h; Δt=Δt)
            end
            pred_busde_arr[m] = T_s
        end

        push!(crps_b_list, crps_ensemble(pred_bude_arr,  y_true))
        push!(crps_s_list, crps_ensemble(pred_busde_arr, y_true))
        push!(crps_c_list, crps_climatology(y_true, T_train))

        q05_b = quantile(pred_bude_arr,  0.05f0); q95_b = quantile(pred_bude_arr,  0.95f0)
        q05_s = quantile(pred_busde_arr, 0.05f0); q95_s = quantile(pred_busde_arr, 0.95f0)
        if q05_b <= y_true <= q95_b; cov_b += 1; end
        if q05_s <= y_true <= q95_s; cov_s += 1; end

        p_b = Float64(mean(pred_bude_arr  .> 0.5f0))
        p_s = Float64(mean(pred_busde_arr .> 0.5f0))
        o   = y_true > 0.5f0 ? 1.0 : 0.0
        push!(brier_b_list, (p_b - o)^2)
        push!(brier_s_list, (p_s - o)^2)

        if H == 6
            push!(probs_bude_6m,  Float32(p_b))
            push!(probs_busde_6m, Float32(p_s))
            push!(obs_events_6m,  Float32(o))
        end
    end

    crps_bude[H]  = mean(crps_b_list);  crps_busde[H] = mean(crps_s_list)
    crps_clima[H] = mean(crps_c_list)
    cov_bude[H]   = cov_b / n_valid;    cov_busde[H]  = cov_s / n_valid
    brier_bude[H] = mean(brier_b_list); brier_busde[H] = mean(brier_s_list)

    @info @sprintf("    CRPS  BUDE=%.3f  BUSDE=%.3f  Clima=%.3f",
                   crps_bude[H], crps_busde[H], crps_clima[H])
    @info @sprintf("    Cov90 BUDE=%4.1f%%  BUSDE=%4.1f%%",
                   cov_bude[H]*100, cov_busde[H]*100)
end

##############################################################################
# 4. Diagrama de confiabilidad a H = 6
##############################################################################

const bins        = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
const bin_centers = [0.1, 0.3, 0.5, 0.7, 0.9]

function compute_reliability(probs, obs)
    freqs  = zeros(5)
    counts = zeros(Int, 5)
    for (p, o) in zip(probs, obs)
        idx = findfirst(b -> b >= p, bins) - 1
        idx = clamp(idx == 0 ? 1 : idx, 1, 5)
        freqs[idx]  += o
        counts[idx] += 1
    end
    for b in 1:5
        freqs[b] = counts[b] > 0 ? freqs[b] / counts[b] : NaN
    end
    return freqs, counts
end

rel_freqs_bude,  rel_counts_bude  = compute_reliability(probs_bude_6m,  obs_events_6m)
rel_freqs_busde, rel_counts_busde = compute_reliability(probs_busde_6m, obs_events_6m)

##############################################################################
# 5. Gráfico de abanico (julio 2014 → 12 meses)
##############################################################################

idx_fan = findfirst(d -> year(d) == 2014 && month(d) == 7, dates_test)
if isnothing(idx_fan); idx_fan = 43; end

H_fan  = 12
M_fan  = 300
n_fan  = min(H_fan + 1, N_test - idx_fan + 1)
t_fan  = t_test[idx_fan : idx_fan + n_fan - 1]
obs_fan = T_test[idx_fan : idx_fan + n_fan - 1]

T0_fan, h0_fan = T_test[idx_fan], h_test[idx_fan]

bude_fan  = fill(NaN32, n_fan, M_fan)
busde_fan = fill(NaN32, n_fan, M_fan)
bude_fan[1,  :] .= T0_fan
busde_fan[1, :] .= T0_fan

for m in 1:M_fan
    ϵ   = ComponentArray(randn(Float32, length(θ_bayes.μ)), getaxes(θ_bayes.μ))
    θ_s = θ_bayes.μ + exp.(θ_bayes.ρ) .* ϵ
    T_b, h_b = T0_fan, h0_fan
    T_s, h_s = T0_fan, h0_fan
    for k in 2:n_fan
        T_b, h_b = bude_step(T_b, h_b, θ_s)
        bude_fan[k, m] = T_b
        T_s, h_s = EnsoModels.busde_step(rng, T_s, h_s, θ_s, snn_f, snn_g, T_scale, h_scale, γ_T, γ_h; Δt=Δt)
        busde_fan[k, m] = T_s
    end
end

bude_q10  = [quantile(bude_fan[k, :], 0.10f0)  for k in 1:n_fan]
bude_q90  = [quantile(bude_fan[k, :], 0.90f0)  for k in 1:n_fan]
bude_mean = [mean(bude_fan[k, :])               for k in 1:n_fan]
busde_q10  = [quantile(busde_fan[k, :], 0.10f0) for k in 1:n_fan]
busde_q90  = [quantile(busde_fan[k, :], 0.90f0) for k in 1:n_fan]
busde_mean = [mean(busde_fan[k, :])              for k in 1:n_fan]

##############################################################################
# 6. Figuras
##############################################################################

const Figure = CairoMakie.Figure
const Axis   = CairoMakie.Axis

col_bude  = RGBf(0.18, 0.53, 0.80)
col_busde = RGBf(0.90, 0.50, 0.13)
col_clim  = RGBf(0.45, 0.45, 0.45)

fig = Figure(size=(1380, 950), fontsize=13)
Hs  = 1:H_max

axA = Axis(fig[1, 1],
    xlabel    = "Tiempo de adelanto (meses)",
    ylabel    = "CRPS (°C)",
    title     = "Panel A - CRPS vs Tiempo de adelanto",
    titlesize = 13)
lines!(axA, Hs, crps_clima; color=col_clim,  linewidth=1.5, linestyle=:dash, label="Climatología")
lines!(axA, Hs, crps_bude;  color=col_bude,  linewidth=2.2, label="BUDE (solo epistémica)")
lines!(axA, Hs, crps_busde; color=col_busde, linewidth=2.2, label="BUSDE (epistémica + aleatoria)")
scatter!(axA, Hs, crps_bude;  color=col_bude,  markersize=7)
scatter!(axA, Hs, crps_busde; color=col_busde, markersize=7)
axislegend(axA, position=:lt, labelsize=11)

axB = Axis(fig[1, 2],
    xlabel    = "Tiempo de adelanto (meses)",
    ylabel    = "Cobertura empírica del PI al 90%",
    title     = "Panel B - Cobertura del Intervalo de Predicción al 90%",
    titlesize = 13)
hlines!(axB, [0.90]; color=:black, linestyle=:dash, linewidth=1.8, label="Ideal (0.90)")
band!(axB, Hs, fill(0.0, H_max), cov_bude;  color=(col_bude,  0.10))
band!(axB, Hs, fill(0.0, H_max), cov_busde; color=(col_busde, 0.10))
lines!(axB, Hs, cov_bude;  color=col_bude,  linewidth=2.2, label="BUDE")
lines!(axB, Hs, cov_busde; color=col_busde, linewidth=2.2, label="BUSDE")
scatter!(axB, Hs, cov_bude;  color=col_bude,  markersize=7)
scatter!(axB, Hs, cov_busde; color=col_busde, markersize=7)
ylims!(axB, 0.0, 1.05)
axislegend(axB, position=:lb, labelsize=11)

axC = Axis(fig[2, 1],
    xlabel    = "P(El Niño | T > 0.5 °C) pronosticada",
    ylabel    = "Frecuencia observada",
    title     = "Panel C - Diagrama de confiabilidad a H = 6 meses\n[tam. marcador ∝ cant. muestras]",
    titlesize = 13)
lines!(axC, [0.0, 1.0], [0.0, 1.0]; color=:black, linestyle=:dash, linewidth=1.2, label="Perfecta")
clim_freq = mean(T_test .> 0.5f0)
lines!(axC, [0.0, 1.0], [clim_freq, clim_freq]; color=(:black, 0.3), linestyle=:dot, linewidth=1.0)

valid_b = findall(!isnan, rel_freqs_bude)
valid_s = findall(!isnan, rel_freqs_busde)
sz_b = [sqrt(rel_counts_bude[b])  * 6 + 7 for b in valid_b]
sz_s = [sqrt(rel_counts_busde[b]) * 6 + 7 for b in valid_s]

scatter!(axC, bin_centers[valid_b], rel_freqs_bude[valid_b]; color=col_bude, markersize=sz_b, label="BUDE")
lines!(axC,  bin_centers[valid_b], rel_freqs_bude[valid_b]; color=col_bude, linewidth=2.0)
scatter!(axC, bin_centers[valid_s], rel_freqs_busde[valid_s]; color=col_busde, markersize=sz_s, label="BUSDE", marker=:diamond)
lines!(axC,  bin_centers[valid_s], rel_freqs_busde[valid_s]; color=col_busde, linewidth=2.0, linestyle=:dash)
xlims!(axC, -0.02, 1.02); ylims!(axC, -0.02, 1.05)
axislegend(axC, position=:lt, labelsize=11)

fan_title = @sprintf("Panel D - Abanico a 12 meses desde Julio 2014\n[BUDE: bandas estrechas; BUSDE: bandas físicamente realistas]")
axD = Axis(fig[2, 2],
    xlabel    = "Año",
    ylabel    = "Anomalía de SST (°C)",
    title     = fan_title,
    titlesize = 13)
band!(axD, t_fan, bude_q10,  bude_q90;  color=(col_bude,  0.25), label="BUDE 10–90%")
band!(axD, t_fan, busde_q10, busde_q90; color=(col_busde, 0.25), label="BUSDE 10–90%")
lines!(axD, t_fan, bude_mean;  color=col_bude,  linewidth=2.0, linestyle=:dash, label="Media BUDE")
lines!(axD, t_fan, busde_mean; color=col_busde, linewidth=2.0, label="Media BUSDE")
lines!(axD, t_fan, obs_fan;    color=:firebrick, linewidth=2.5, label="Observado")
hlines!(axD, [0.5f0];  color=(:firebrick, 0.4),  linestyle=:dot, linewidth=1.0)
hlines!(axD, [-0.5f0]; color=(:dodgerblue, 0.4), linestyle=:dot, linewidth=1.0)
axislegend(axD, position=:lt, labelsize=11)

save("figures/07_bude_vs_busde.png", fig; px_per_unit=2)
@info "Guardado figures/07_bude_vs_busde.png"

##############################################################################
# 7. Tabla de resumen
##############################################################################

@info "===== Resumen de Habilidad Probabilística BUDE vs BUSDE ====="
@info @sprintf("%-5s  %-14s %-14s %-13s  %-13s  %-13s",
               "H", "CRPS_BUDE", "CRPS_BUSDE", "Cov_BUDE", "Cov_BUSDE", "Clima_CRPS")
for H in 1:H_max
    @info @sprintf("H=%2d   %.3f          %.3f          %5.1f%%        %5.1f%%         %.3f",
        H, crps_bude[H], crps_busde[H],
        cov_bude[H]*100, cov_busde[H]*100,
        crps_clima[H])
end
