# Correspondencia con la guía del curso

Qué pide cada sección de la guía del proyecto final (*Proyecto Final HPC 2026-2*, capítulos
1, 5 y 6) y dónde se cumple en este repositorio.

## 1. Requisitos y dónde se cumplen

| Sección de la guía | Qué pide | Dónde está |
|---|---|---|
| §1.2 (1) | Python puro, referencia de corrección | `python main.py pure` · `python/adapters/utils_pure.py` |
| §1.2 (2) | NumPy vectorizado | `python main.py numpy` · `python/adapters/utils_npcp.py` |
| §1.2 (3), §6.2 | Paralelo en CPU con multiprocessing/joblib/Dask, con el nº de trabajadores como parámetro | `python main.py multiprocessing --workers p` (réplicas: `utils_parallel.py`; franjas: `utils_mp.py`) |
| §1.2 (4), §6.2 | CuPy, migración de NumPy, midiendo aparte copia a GPU, cómputo y copia de vuelta | `python main.py cupy` (mismo `utils_npcp.py`); tiempos en `outputs/gpu/times_gpu.csv` |
| §1.2 (5–7), §6.3 | C++ secuencial, OpenMP y MPI con Makefile | `cpp/secuencial.cpp`, `cpp/openmp.cpp`, `cpp/mpi.cpp`, `cpp/Makefile` |
| §1.2 (8), §6.2 | Kernel CUDA opcional (nvcc o `cupy.RawKernel`) | `python main.py cuda` · `python/adapters/kernel_automata.cu` + `utils_cuda.py` |
| §1.5 | GPU en Colab; no usar más procesos que núcleos físicos; indicar el hardware | `colab/GPU_en_Colab.ipynb`; `tools/run_all.sh` detecta los núcleos físicos; `outputs/hardware.txt` y pie de cada figura |
| §5.6 paso 5 | multiprocessing al nivel de las realizaciones | Estrategia por defecto `--estrategia replicas` |
| §5.6 paso 8 | MPI: reparto y, como extensión, descomposición de dominio con `MPI_Sendrecv`; comparar esquemas | `cpp/mpi.cpp` (franjas con filas fantasma); comparación réplicas/franjas en multiprocessing |
| §5.7 | Validación estadística (bandas de confianza, tamaño final dentro del error estándar); semillas distintas; doble buffer | `tools/compare_stats.py`, `tools/validate_metrics.py`; figura G2; tabla 4 del informe |
| §5.8 | Gráficas propias y actualizaciones de celda por segundo | Notebook: P2 (actualizaciones/s), P3 (variabilidad entre réplicas) |
| §6.1 | Estructura del repositorio; notebook que genera TODAS las figuras | `notebooks/analisis.ipynb` → `outputs/figures/` (ver nombres abajo) |
| §6.3 | Hilos con `OMP_NUM_THREADS`; justificar el scheduling | `./openmp --threads p` u `OMP_NUM_THREADS`; `--schedule`; figura P4b |
| §6.4 | Medir solo el cómputo; calentamiento; ≥ 5 repeticiones; sincronización en GPU y MPI; semillas documentadas | `--warmup`, `--runs 5`; eventos CUDA; `MPI_Barrier` + máximo; semillas en README |
| §6.4 | `tiempos.csv` con implementación, tamaño, trabajadores, repetición, tiempos, precisión y equipo | `outputs/tiempos.csv` (lo deriva el notebook de `outputs/times.csv` y `outputs/gpu/`) |
| §6.5 | Speedup frente a NumPy y a C++ secuencial, eficiencia, Karp-Flatt, Amdahl, rendimiento, fracción de transferencia | Notebook (tablas `tabla_escalabilidad.csv`, `tabla_gpu.csv`) e informe §7 |
| §6.6 | Gráficas G1–G7 (G8 recomendada) | G1–G7 en el notebook; G8 (float32 vs float64) no se hizo |
| §6.7 | Informe PDF de 12–20 páginas que responda las 8 preguntas | `informe.pdf` (§8.1 responde las 8 preguntas) |
| §6.8 | Sustentación con demo de CuPy y de una versión compilada; ejecuciones documentadas | `docs/presentacion.pptx`; demo en `GUIA.md` §8; comandos en `docs/CHECKLIST_REVISION.md` |

## 2. Nombres de la guía y nombres de este repositorio

La guía sugiere una estructura de archivos; este repositorio la cumple con otros nombres
(un único `main.py` con subcomandos, que reutiliza la carga de datos y el formato de salida):

| Guía (§6.1) | Este repositorio |
|---|---|
| `python/comun.py` | `python/domain/constants.py`, `python/domain/enums.py` |
| `python/01_secuencial.py` | `python main.py pure` |
| `python/02_numpy.py` | `python main.py numpy` |
| `python/03_paralelo_cpu.py` | `python main.py multiprocessing` |
| `python/04_cupy.py` | `python main.py cupy` |
| `python/05_cupy_rawkernel.py` | `python main.py cuda` |
| `cpp/*.cpp`, `cpp/Makefile` | igual |
| `resultados/tiempos.csv` | `outputs/times.csv` (+ `outputs/tiempos.csv` con las columnas de la guía) |
| `resultados/validacion/` | `outputs/metrics/` + `outputs/validation_summary.csv` |
| `resultados/figuras/` | `outputs/figures/` |
| `notebooks/analisis.ipynb` | igual |
| `informe.pdf` | igual (generado desde `docs/informe_plantilla.md`) |

## 3. Decisiones propias y su justificación

1. **Un flujo aleatorio por fila en C++.** Secuencial, OpenMP y MPI producen el mismo
   resultado para la misma semilla con cualquier número de hilos o procesos, y la
   corrección del paralelismo se comprueba con `cmp`. Cumple §6.4 ("flujos independientes
   por proceso, hilo o realización").
2. **Binomial exacta en todas las versiones.** La aproximación original de NumPy/CuPy sesgaba
   el modelo cerca del umbral epidémico (informe §6.2); se reemplazó por la exacta
   vectorizada. El kernel CUDA implementa su propia binomial exacta (inversión + BTRS).
3. **`lgamma` reentrante en C++.** Evita la contención entre hilos de OpenMP (informe §7.8,
   pregunta 7 de §6.7).
4. **GPU en Colab, resultados aparte.** Se guardan en `outputs/gpu/` para no mezclar dos
   equipos; el notebook los une y cada figura indica en qué hardware se midió cada parte.
5. **`--board-size` en los binarios C++.** Solo para la escalabilidad débil de MPI (G6); sus
   tiempos van a `outputs/experiments/weak/`, no a `times.csv`.
6. **Columnas de las métricas.** `AGENTS.md` habla de "18 columnas", pero la tabla lista 19
   nombres (`paradigm, region, fecha` + 16); se usa el formato real.
