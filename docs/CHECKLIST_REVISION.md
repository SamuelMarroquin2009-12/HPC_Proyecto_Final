# Lista de revisión

Comandos exactos para reproducir todo, qué debe salir en cada uno y de dónde sale cada número
del informe. Todos los comandos se ejecutan **desde la raíz del repositorio**, en WSL/Linux,
con el entorno activado (`source .venv/bin/activate`). La preparación está en `GUIA.md` §2.

## 1. Todo con un comando

| Comando | Qué hace | Debe terminar con |
|---|---|---|
| `RAPIDO=1 bash tools/run_all.sh` | Prueba rápida (~5 min): solo Amazonía, 2 repeticiones | `LISTO` |
| `bash tools/run_all.sh` | Medición completa de CPU (~1,5 h en 4 núcleos) + notebook + PDF + presentación | `LISTO` |
| `bash tools/run_all.sh --solo-informe` | No mide: valida y regenera notebook, PDF y presentación (tras copiar `outputs/gpu/`) | `LISTO` |
| Colab: `colab/GPU_en_Colab.ipynb` → Ejecutar todas | CuPy y kernel CUDA en GPU T4; descarga `resultados_gpu.zip` | se descarga el zip |

## 2. Comprobaciones paso a paso

**Compilación sin advertencias**

```bash
make -C cpp clean all
```
Debe compilar `secuencial`, `openmp` y `mpi` sin ninguna línea `warning`.

**El paralelismo no cambia el resultado (C++)**

```bash
cd cpp && ./secuencial --region pacifico --warmup 0 --outdir /tmp/chk && ./openmp --region pacifico --threads 4 --warmup 0 --outdir /tmp/chk && mpirun -np 4 ./mpi --region pacifico --warmup 0 --outdir /tmp/chk && cd ..
```

```bash
for p in openmp mpi; do cmp <(cut -d, -f2- /tmp/chk/metrics/serial_pacifico.csv) <(cut -d, -f2- /tmp/chk/metrics/${p}_pacifico.csv) && echo "$p: idéntico a serial"; done
```
Debe imprimir `openmp: idéntico a serial` y `mpi: idéntico a serial` (L = 87 no es divisible
por 4, así que también prueba el reparto de filas sobrantes en MPI).

**Muestreador binomial de NumPy/CuPy**

```bash
python tools/test_fast_binomial.py
```
Debe terminar en `OK`.

**Kernel CUDA sin GPU (emulado en CPU)**

```bash
python tools/test_kernel_cpu.py --region amazonia --runs 8
```
Debe mostrar `8/8 archivos válidos` y la comparación con Python puro.

**Invariantes de todos los resultados**

```bash
python tools/validate_metrics.py
```
Debe terminar con `N/N archivos válidos`.

**Validación estadística entre implementaciones**

```bash
python tools/compare_stats.py
```
Escribe `outputs/validation_summary.csv`. Esperado: todas las implementaciones con la curva
media dentro de la banda la mayoría de los meses y el tamaño final dentro de 2 errores
estándar; la excepción documentada es C++ en algún caso puntual, por el sesgo de ~0,04 % de
`std::binomial_distribution` (informe §6.2; `tools/binomial_bias_test.cpp`).

## 3. De dónde sale cada número del informe

Todas las cifras del informe y de la presentación las inserta `tools/resultados.py` a partir de
las tablas que escribe el notebook en `outputs/figures/`. Para ver todos los valores:

```bash
python tools/resultados.py
```

| Dato del informe | Tabla / figura | Origen en los datos |
|---|---|---|
| Tiempos por región (Tabla 5) | `tabla_mejor_por_region.csv`, P1 | mediana de `outputs/times.csv` (+ `outputs/gpu/times.csv`) por paradigma y región |
| Speedup, eficiencia, Karp-Flatt, Amdahl (Tabla 6) | `tabla_escalabilidad.csv`, G5 | `openmp`/`mpi` ÷ `serial`; `multiprocessing` ÷ su `workers=1` (mismo lote de 8 réplicas); `multiprocessing_franjas` ÷ su `workers=1` |
| Speedup máximo (Resumen, §7.3) | G4 | mejor configuración sin sobresuscripción |
| Pendientes log-log (§7.2) | `tabla_pendientes.csv`, G3 | ajuste de log(t) contra log(L) |
| Escalabilidad débil (Tabla 7) | `tabla_escalabilidad_debil.csv`, G6 | `outputs/experiments/weak/times.csv` |
| GPU: transferencias y cómputo (Tabla 9b) | `tabla_gpu.csv`, G7 | `outputs/gpu/times_gpu.csv` (Colab) |
| Tamaño de cruce GPU/CPU (§7.3, §8.1 p. 4) | G4 | primer L en que la mejor GPU es más rápida que la mejor CPU para una simulación |
| Efecto de `lgamma` (Tabla 8) | P4a | `outputs/experiments/times_cpp_glibc_lgamma.csv` frente a `outputs/times.csv` |
| Scheduling (Tabla 9) | P4b | `outputs/experiments/schedule/schedule.csv` |
| Validación (Tabla 4) | `tabla_validacion.csv`, G2 | réplicas `*_run*.csv` frente a `pure_*_run*.csv` |
| Variabilidad entre réplicas (§6.1) | `tabla_variabilidad.csv`, P3 | réplicas `*_run*.csv` |
| Hardware (§5) | pie de cada figura | `outputs/hardware.txt` y `outputs/gpu/hardware_gpu.txt` |

## 4. Pendientes antes de entregar

- [ ] Medición completa en el portátil: `bash tools/run_all.sh`.
- [ ] GPU en Colab y `bash tools/run_all.sh --solo-informe`.
- [ ] Completar lo marcado en amarillo en `informe.pdf` (horas de trabajo, referencias por
      verificar, medición con perfilador) editando `docs/informe_plantilla.md`.
- [ ] Subir resultados, PDF y presentación al repositorio (`GUIA.md` §5).
- [ ] Ensayar la demostración en vivo (`GUIA.md` §8) y estudiar las 15 preguntas de
      `docs/presentacion.md`.
