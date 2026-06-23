##############################################################################
# 02_baseline.jl
# Proyecto ENSO — Línea de base del oscilador de recarga lineal
#
# Modelo (Jin 1997, linealizado):
#   dT/dt = a*T + b*h
#   dh/dt = -c*T + d*h       con restricción d = -a  (traza cero)
#
# Ajuste: OLS sobre aproximación de derivadas por diferencias finitas.
#
# Entradas: data/enso_dataset.csv
# Salidas:  data/baseline_params.json
#           figures/02_baseline.png
##############################################################################

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using DataFrames, CSV, Dates, LinearAlgebra, Statistics
using DifferentialEquations, CairoMakie, JSON3

cd(@__DIR__)
mkpath("data"); mkpath("figures")

##############################################################################
# 1. Cargar datos
##############################################################################

df_all = CSV.read("data/enso_dataset.csv", DataFrame)

mask  = (year.(df_all.date) .>= 1980) .& (year.(df_all.date) .<= 2010)
df    = df_all[mask, :]
T     = df.T_anomaly   # Anomalía de SST (°C)
h     = df.h_anomaly   # Anomalía de WWV (10¹⁴ m³)
dates = df.date

t_num = year.(dates) .+ (month.(dates) .- 1) ./ 12
dt    = 1.0 / 12.0   # años

N = length(T)

##############################################################################
# 2. Ajustar parámetros mediante OLS sobre diferencias finitas
#
#    Diferencias centrales para puntos interiores:
#      dT_i ≈ (T[i+1] - T[i-1]) / (2Δt)
#      dh_i ≈ (h[i+1] - h[i-1]) / (2Δt)
#
#    Se ajustan 3 parámetros libres (a, b, c); d = -a por construcción.
##############################################################################

idx = 2:(N-1)

dT = (T[idx .+ 1] .- T[idx .- 1]) ./ (2dt)
dh = (h[idx .+ 1] .- h[idx .- 1]) ./ (2dt)

T_c = T[idx]
h_c = h[idx]

# OLS para la ecuación de T: dT = a*T + b*h
XT = hcat(T_c, h_c)
θT = XT \ dT
a, b = θT[1], θT[2]
d    = -a   # restricción de traza cero

# OLS para la ecuación de h: dh = -c*T + d*h
# → dh + a*h = -c*T  → regresión de (dh + a*h) sobre T
XH = -T_c
θH = XH \ (dh .+ a .* h_c)
c  = θH[1]

# Período natural: λ = a ± i√(bc), T_nat = 2π/√(bc)
if b * c <= 0
    @warn "b*c = $(b*c) ≤ 0 → el sistema no es oscilatorio; T_nat = NaN"
    ω₀    = NaN
    T_nat = NaN
else
    ω₀    = sqrt(b * c)
    T_nat = 2π / ω₀
end

@info "Parámetros ajustados:"
@info "  a = $(round(a, digits=4)) año⁻¹"
@info "  b = $(round(b, digits=4)) año⁻¹"
@info "  c = $(round(c, digits=4)) año⁻¹"
@info "  d = -a = $(round(d, digits=4)) año⁻¹"
@info "  Período natural: $(round(T_nat, digits=2)) años"

params = Dict("a"=>a, "b"=>b, "c"=>c, "d"=>d, "T_nat_yr"=>T_nat)
open("data/baseline_params.json", "w") do f
    JSON3.write(f, params)
end

##############################################################################
# 3. Simular la ODE lineal con los parámetros ajustados
##############################################################################

function recharge_oscillator!(du, u, p, t)
    T, h = u
    a, b, c, d = p
    du[1] = a*T + b*h
    du[2] = -c*T + d*h
end

p   = [a, b, c, d]
u0  = [T[1], h[1]]
tspan = (t_num[1], t_num[end])

prob = ODEProblem(recharge_oscillator!, u0, tspan, p)
sol  = solve(prob, Tsit5(), saveat=dt)

T_sim = [u[1] for u in sol.u]
h_sim = [u[2] for u in sol.u]
t_sim = sol.t

##############################################################################
# 4. Métricas
##############################################################################

n_plot = min(N, length(t_sim))

rmse_T = sqrt(mean((T[1:n_plot] .- T_sim[1:n_plot]).^2))
corr_T = cor(T[1:n_plot], T_sim[1:n_plot])

@info "RMSE de línea de base (T): $(round(rmse_T, digits=3)) °C"
@info "Corr de línea de base (T): $(round(corr_T, digits=3))"

##############################################################################
# 5. Figura
##############################################################################

fig = CairoMakie.Figure(size=(1000, 750), fontsize=13)

ax1 = CairoMakie.Axis(fig[1,1],
    xlabel="Año",
    ylabel="Anomalía de SST (°C)",
    title="Oscilador de recarga lineal — SST (T)")
lines!(ax1, t_num, T, color=:firebrick, linewidth=1.5, label="Observado")
lines!(ax1, t_sim[1:n_plot], T_sim[1:n_plot],
    color=:dodgerblue, linewidth=1.2, linestyle=:dash, label="ODE de línea de base")
axislegend(ax1, position=:lt)

ax2 = CairoMakie.Axis(fig[2,1],
    xlabel="Año",
    ylabel="Anomalía de WWV (10¹⁴ m³)",
    title="Oscilador de recarga lineal — Profundidad de la termoclina (h)")
lines!(ax2, t_num, h, color=:steelblue, linewidth=1.5, label="Observado")
lines!(ax2, t_sim[1:n_plot], h_sim[1:n_plot],
    color=:orange, linewidth=1.2, linestyle=:dash, label="ODE de línea de base")
axislegend(ax2, position=:lt)

ax3 = CairoMakie.Axis(fig[3,1],
    xlabel="Anomalía de T (°C)",
    ylabel="Anomalía de h (10¹⁴ m³)",
    title="Espacio de fases  [RMSE = $(round(rmse_T,digits=3)) °C,  r = $(round(corr_T,digits=3))]")
scatter!(ax3, T, h, color=:firebrick, markersize=3, alpha=0.6, label="Observado")
lines!(ax3, T_sim[1:n_plot], h_sim[1:n_plot],
    color=:dodgerblue, linewidth=1.5, label="ODE de línea de base")
axislegend(ax3, position=:lt)

save("figures/02_baseline.png", fig, px_per_unit=2)
@info "Guardado figures/02_baseline.png"