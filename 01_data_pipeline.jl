##############################################################################
# 01_data_pipeline.jl
# Proyecto ENSO - Adquisición y preprocesamiento de datos
#
# Salidas:
#   data/enso_dataset.csv
#   data/T_anomaly.csv
#   data/h_anomaly.csv
#   figures/01_exploratory.png
##############################################################################

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using NCDatasets, Statistics, Dates, DataFrames, CSV, CairoMakie, Downloads

cd(@__DIR__)
mkpath("data"); mkpath("figures")

nanmean(v) = mean(filter(!isnan, v))

##############################################################################
# 1. WWV - Anomalía del Volumen de Agua Cálida (proxy h)
#    Fuente: Índice WWV de NOAA PMEL (1980–2021).
##############################################################################

const WWV_RAW_FILE = "data/wwv_raw.txt"
wwv_raw_content = read(WWV_RAW_FILE, String)

# Formato del archivo: AAAAMM  Volumen  Anomalía
function parse_wwv_inline(raw::String)
    dates  = Date[]
    anom   = Float64[]
    for line in split(raw, '\n')
        s = strip(line)
        isempty(s) && continue
        parts = split(s)
        length(parts) < 3 && continue
        yyyymm = tryparse(Int, parts[1]); isnothing(yyyymm) && continue
        yr = yyyymm ÷ 100
        mo = yyyymm % 100
        (mo < 1 || mo > 12) && continue
        v = tryparse(Float64, parts[3]); isnothing(v) && continue
        push!(dates, Date(yr, mo, 1))
        push!(anom,  v)
    end
    return dates, anom
end

wwv_dates_all, h_anom_all = parse_wwv_inline(wwv_raw_content)

mask_wwv  = (year.(wwv_dates_all) .>= 1980) .& (year.(wwv_dates_all) .<= 2021)
wwv_dates = wwv_dates_all[mask_wwv]
h_anom    = h_anom_all[mask_wwv]

h_anom = h_anom ./ 1e14   # m³ → 10¹⁴ m³

# Recentrar respecto a la media del periodo de entrenamiento
train_mask   = (year.(wwv_dates) .>= 1980) .& (year.(wwv_dates) .<= 2010)
h_mean_train = mean(h_anom[train_mask])
h_anom       = h_anom .- h_mean_train

@info "WWV: $(length(h_anom)) valores,  $(wwv_dates[1]) – $(wwv_dates[end])"

##############################################################################
# 2. SST - ERSSTv5
#    Fuente: NOAA Extended Reconstructed SST V5.
#    Región Niño 3.4: 5°S–5°N, 170°W–120°W.
#    Climatología de referencia: 1980–2010 (sin fuga al periodo de test).
##############################################################################

const SST_URL  = "https://raw.githubusercontent.com/pydata/xarray-data/master/ersstv5.nc"
const SST_FILE = "data/ersstv5.nc"

if !isfile(SST_FILE)
    @info "Descargando ERSSTv5..."
    Downloads.download(SST_URL, SST_FILE)
end

T_dates = Date[]
T_anom  = Float64[]

NCDataset(SST_FILE, "r") do ds
    sst_var  = ds["sst"]
    lat      = Array(ds["lat"])
    lon      = Array(ds["lon"])
    time_raw = Array(ds["time"])

    @info "Tamaño de sst: $(size(sst_var))  dims: $(dimnames(sst_var))"

    sst_full = Float64.(coalesce.(Array(sst_var), NaN))

    dims    = dimnames(sst_var)
    lon_ax  = findfirst(==("lon"),  dims)
    lat_ax  = findfirst(==("lat"),  dims)
    time_ax = findfirst(==("time"), dims)

    time_mask = (year.(time_raw) .>= 1970) .& (year.(time_raw) .<= 2021)
    lat_mask  = (-6 .<= lat .<= 6)
    lon_mask  = (190 .<= lon .<= 240)

    t_idx = findall(time_mask)
    la_idx = findall(lat_mask)
    lo_idx = findall(lon_mask)
    time_sel = time_raw[time_mask]

    idx          = Vector{Any}(undef, 3)
    idx[lon_ax]  = lo_idx
    idx[lat_ax]  = la_idx
    idx[time_ax] = t_idx

    sst_region = sst_full[idx...]
    N_t        = length(t_idx)

    T_raw = Vector{Float64}(undef, N_t)
    for ti in 1:N_t
        si          = Vector{Any}(undef, 3)
        si[lon_ax]  = Colon()
        si[lat_ax]  = Colon()
        si[time_ax] = ti
        T_raw[ti]   = nanmean(vec(sst_region[si...]))
    end

    clim_mask   = (year.(time_sel) .>= 1980) .& (year.(time_sel) .<= 2010)
    months_clim = month.(time_sel[clim_mask])
    clim        = [nanmean(T_raw[clim_mask][months_clim .== m]) for m in 1:12]

    months = month.(time_sel)
    anom   = T_raw .- clim[months]

    append!(T_dates, [Date(year(t), month(t), 1) for t in time_sel])
    append!(T_anom,  anom)
end

@info "SST: $(length(T_anom)) valores,  $(T_dates[1]) – $(T_dates[end])"

##############################################################################
# 3. Alinear y guardar
##############################################################################

T_dict = Dict(zip(T_dates, T_anom))
h_dict = Dict(zip(wwv_dates, h_anom))
common = sort(collect(intersect(Set(T_dates), Set(wwv_dates))))

T_aligned = [T_dict[d] for d in common]
h_aligned = [h_dict[d] for d in common]

df = DataFrame(date=common, T_anomaly=T_aligned, h_anomaly=h_aligned)
CSV.write("data/enso_dataset.csv", df)
CSV.write("data/T_anomaly.csv",    select(df, :date, :T_anomaly))
CSV.write("data/h_anomaly.csv",    select(df, :date, :h_anomaly))
@info "Guardado data/enso_dataset.csv  ($(nrow(df)) filas, $(common[1]) – $(common[end]))"

##############################################################################
# 4. Detección de eventos de El Niño
#    Criterio ONI (NOAA): SST > 0.5 °C durante al menos 5 meses consecutivos.
##############################################################################

function detect_el_nino_peaks(T, dates; threshold=0.5, min_duration=5)
    peaks = Date[]; n = length(T); i = 1
    while i <= n
        if T[i] > threshold
            j = i
            while j <= n && T[j] > threshold; j += 1; end
            if j - i >= min_duration
                push!(peaks, dates[i + argmax(T[i:j-1]) - 1])
            end
            i = j
        else
            i += 1
        end
    end
    return peaks
end

peaks        = detect_el_nino_peaks(T_aligned, common)
t_peaks      = year.(peaks) .+ (month.(peaks) .- 1) ./ 12
intervals_yr = diff(t_peaks)
cv           = std(intervals_yr) / mean(intervals_yr)

@info "Eventos de El Niño: $(length(peaks))  |  CV = $(round(cv, digits=2))"

##############################################################################
# 5. Figura exploratoria
##############################################################################

t_num = year.(common) .+ (month.(common) .- 1) ./ 12
fig   = Figure(size=(1000, 750), fontsize=13)

ax1 = Axis(fig[1,1], xlabel="Año", ylabel="Anomalía de SST (°C)",
           title="Anomalía de SST en Niño 3.4 - ERSSTv5 (1980–2021)")
lines!(ax1, t_num, T_aligned; color=:firebrick, linewidth=1.3)
hlines!(ax1, [0.5]; color=:gray, linestyle=:dash)
for pk in peaks
    vlines!(ax1, [year(pk) + (month(pk)-1)/12]; color=(:orange, 0.4), linewidth=1)
end

ax2 = Axis(fig[2,1], xlabel="Año", ylabel="Anomalía de WWV (10¹⁴ m³)",
           title="Anomalía del Volumen de Agua Cálida - proxy h (PMEL)")
lines!(ax2, t_num, h_aligned; color=:steelblue, linewidth=1.3)
hlines!(ax2, [0.0]; color=:gray, linestyle=:dash)

ax3 = Axis(fig[3,1], xlabel="Intervalo (años)", ylabel="Frecuencia",
           title="Intervalos entre eventos  [CV = $(round(cv, digits=2))]")
hist!(ax3, intervals_yr; bins=10, color=:slateblue,
      strokecolor=:white, strokewidth=1)

save("figures/01_exploratory.png", fig; px_per_unit=2)
@info "Guardado figures/01_exploratory.png"