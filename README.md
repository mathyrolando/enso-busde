# enso-busde

Pronóstico estocástico de ENSO mediante una **Ecuación Diferencial Universal Bayesiana** (BUDE) extendida con un término de difusión dependiente del estado (BUSDE).

Proyecto final - Datos, Ecuaciones Diferenciales e Inteligencia Artificial

---

## Idea central

El [oscilador de recarga de Jin (1997)](https://doi.org/10.1175/1520-0442(1997)010<0301:AUITSO>2.0.CO;2) describe la dinámica de ENSO como un sistema de dos variables:

- **T**: anomalía de temperatura superficial del mar (SST) en la región Niño 3.4
- **h**: anomalía del volumen de agua cálida (proxy de la profundidad de la termoclina)

Este proyecto aprende las correcciones no lineales al modelo lineal directamente de los datos, combinando tres capas de modelado:

1. **OLS**: ajuste de los parámetros lineales del oscilador por mínimos cuadrados
2. **BUDE**: UDE bayesiana: los parámetros lineales y las redes neuronales correctoras se optimizan conjuntamente mediante Bayes-by-Backprop (ELBO). La incertidumbre epistémica se propaga en el ensemble de predicciones.
3. **BUSDE**: extiende la BUDE con un término de ruido heterocedástico ajustado por MLE, que representa el forzamiento atmosférico irreducible (ráfagas de viento del oeste).

---

## Estructura del repositorio

```
.
├── 01_data_pipeline.jl        # Descarga y preprocesamiento de SST (ERSSTv5) y WWV (PMEL)
├── 02_baseline.jl             # Ajuste OLS del oscilador lineal
├── 03_ude.jl                  # Entrenamiento de la BUDE (variacional)
├── 04_evaluation.jl           # Evaluación fuera de muestra (2011–2021), skill scores
├── 05_sde_stochastic.jl       # Ajuste de ruido SDE y simulaciones a 500 años
├── 06_probabilistic_forecast.jl  # Pronósticos probabilísticos y métricas (CRPS, Brier)
├── 07_bude_vs_busde.jl        # Comparación calibración BUDE vs BUSDE
├── 08_crossval_rolling.jl     # Validación cruzada rolling-origin (5 folds)
├── run.jl                     # Ejecuta el pipeline completo en orden
├── src/
│   └── EnsoModels.jl          # Módulo: integradores, redes, funciones de pérdida, métricas
├── data/
│   └── wwv_raw.txt            # Índice WWV de NOAA PMEL (dato crudo, no regenerable)
├── figures/                   # Figuras generadas por cada script
├── Project.toml
└── Manifest.toml
```

---

## Datos

| Variable | Fuente | Período |
|---|---|---|
| SST (Niño 3.4) | [NOAA ERSSTv5](https://psl.noaa.gov/data/gridded/data.noaa.ersst.v5.html) | 1970–2021 |
| WWV (proxy h) | [NOAA PMEL](https://www.pmel.noaa.gov/tao/wwv/data/) | 1980–2021 |

El archivo NetCDF de ERSSTv5 se descarga automáticamente en el primer paso. El índice WWV (`data/wwv_raw.txt`) está incluido en el repositorio porque no tiene una URL de descarga directa estable.

- **Entrenamiento**: 1980–2010
- **Test fuera de muestra**: 2011–2021

---

## Cómo ejecutar

Requiere [Julia ≥ 1.9](https://julialang.org/downloads/).

```julia
# Desde la carpeta raíz del proyecto
julia run.jl
```

O bien, cada script puede ejecutarse de forma independiente en orden (cada uno activa el entorno y carga sus dependencias).

Las dependencias se instalan automáticamente en el primer run via `Pkg.instantiate()`.

---

## Dependencias principales

`Lux` · `Zygote` · `Optimisers` · `DifferentialEquations` · `StochasticDiffEq` · `ComponentArrays` · `CairoMakie` · `NCDatasets`
