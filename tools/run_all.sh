#!/usr/bin/env bash
# =============================================================================
# Ejecuta TODO el proyecto en este computador, de principio a fin:
#   datos -> compilación -> mediciones -> validación -> notebook -> informe.pdf -> presentación
#
# Uso (desde la raíz del repo, con el entorno de Python activado; ver GUIA.md):
#   bash tools/run_all.sh                  # medición completa (~1,5 h en un portátil de 4 núcleos)
#   RAPIDO=1 bash tools/run_all.sh         # prueba rápida (~5 min): solo Amazonía, 2 repeticiones
#   bash tools/run_all.sh --solo-informe   # NO mide nada: solo valida y regenera notebook, PDF y presentación
#                                          # (úselo después de copiar los resultados de GPU de Colab a outputs/gpu/)
#
# Variables opcionales:
#   RUNS=5                 repeticiones por configuración (la guía pide al menos 5)
#   THREADS="1 2 4 8"      hilos de OpenMP / procesos de multiprocessing (por defecto: 1, 2, 4... hasta los núcleos
#                          físicos, más los hilos lógicos como punto "sobresuscrito" de comparación)
#   PROCS="1 2 4"          procesos de MPI (por defecto: hasta los núcleos físicos)
#   LIMPIAR=1              borra los resultados de CPU anteriores antes de medir (nunca borra outputs/gpu)
#
# Las versiones de GPU (CuPy y kernel CUDA) se miden en Google Colab: colab/GPU_en_Colab.ipynb.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
. tools/_equipo.sh
PY=${PY:-python3}; PYTOOLS=${PYTOOLS:-python3}; RUNS=${RUNS:-5}; LIMPIAR=${LIMPIAR:-1}
paso() { echo; echo "=== [$(date +%H:%M:%S)] $* ==="; }

# --- 0. Requisitos -------------------------------------------------------------
falta=""
for cmd in g++ make mpicxx mpirun; do command -v "$cmd" >/dev/null 2>&1 || falta="$falta $cmd"; done
"$PYTOOLS" -c "import pandas, numpy, matplotlib, pyarrow, nbformat, nbclient, pptx" 2>/dev/null || falta="$falta (paquetes-de-Python)"
if [ -n "$falta" ]; then
  echo "Faltan requisitos:$falta"
  echo "Instálelos con los comandos de GUIA.md, sección 'Preparación (una sola vez)'."
  exit 1
fi
command -v pandoc >/dev/null 2>&1 || echo "AVISO: falta pandoc; todo funciona pero no se generará informe.pdf (sudo apt install pandoc)."

regenerar_informe() {
  paso "Validación"
  $PYTOOLS tools/validate_metrics.py | tail -1
  $PYTOOLS tools/compare_stats.py > /dev/null || true   # código 2 = algún criterio de 2 SE no se cumple (ver informe §6.2)
  paso "Notebook, informe y presentación"
  $PYTOOLS -m jupyter execute --inplace notebooks/analisis.ipynb 2>&1 | grep -iE "error|exception" || true
  $PYTOOLS tools/build_informe.py
  $PYTOOLS tools/build_presentacion.py
  paso "LISTO. Revise outputs/figures/, informe.pdf y docs/presentacion.pptx"
}

if [ "${1:-}" = "--solo-informe" ]; then
  regenerar_informe
  exit 0
fi

# --- Workers según el equipo -----------------------------------------------------
FIS=$(nucleos_fisicos); LOG=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo "$FIS")
lista_hasta() { local out="" w; for w in 1 2 4 8 12 16 24 32; do [ "$w" -le "$1" ] && out="$out $w"; done
                case " $out " in *" $1 "*) ;; *) out="$out $1";; esac; echo $out; }
PROCS=${PROCS:-$(lista_hasta "$FIS")}
if [ -z "${THREADS:-}" ]; then
  THREADS=$(lista_hasta "$FIS")
  [ "$LOG" -gt "$FIS" ] && THREADS="$THREADS $LOG"   # hilos lógicos: punto sobresuscrito (se marca aparte)
fi
MAXF=$(echo $PROCS | tr ' ' '\n' | sort -n | tail -1)
REG_ESCALA="amazonia pacifico"; REG_RESTO="orinoquia caribe andina"; PURE_A=24; PURE_P=8; REG_FRANJAS="amazonia pacifico"
if [ "${RAPIDO:-0}" = "1" ]; then RUNS=2; REG_ESCALA="amazonia"; REG_RESTO=""; PURE_A=4; PURE_P=0; REG_FRANJAS="amazonia"; fi

paso "0. Equipo: $FIS núcleos físicos, $LOG hilos lógicos | OpenMP/multiprocessing: $THREADS | MPI: $PROCS | repeticiones: $RUNS"
mkdir -p outputs/metrics outputs/animations outputs/experiments
if [ "$LIMPIAR" = "1" ]; then
  rm -f outputs/times.csv outputs/tiempos.csv outputs/validation_summary.csv outputs/metrics/*.csv
  rm -rf outputs/experiments/weak outputs/experiments/schedule outputs/experiments/times_cpp_glibc_lgamma.csv
fi
bash tools/hardware_info.sh > outputs/hardware.txt; cat outputs/hardware.txt

paso "1. Datos y compilación"
$PYTOOLS tools/export_csv.py
make -C cpp -s clean all openmp_glibc_lgamma

paso "2. C++: secuencial, OpenMP y MPI"
RUNS=$RUNS THREADS="$THREADS" PROCS="$PROCS" bash tools/benchmark_cpp.sh $REG_ESCALA
[ -n "$REG_RESTO" ] && RUNS=$RUNS THREADS="$MAXF" PROCS="$MAXF" bash tools/benchmark_cpp.sh $REG_RESTO

paso "3. Experimentos: escalabilidad débil, scheduling y lgamma"
RUNS=$RUNS MAXP=$MAXF bash tools/benchmark_weak.sh
RUNS=$RUNS THREADS=$MAXF bash tools/benchmark_schedule.sh $REG_ESCALA
TMP=outputs/experiments/_lgamma_tmp; rm -rf "$TMP"
for region in $REG_ESCALA; do for t in $(lista_hasta "$FIS"); do
  (cd cpp && ./openmp_glibc_lgamma --region "$region" --threads "$t" --runs "$RUNS" --no-metrics --outdir "../$TMP" | grep elapsed)
done; done
mv "$TMP/times.csv" outputs/experiments/times_cpp_glibc_lgamma.csv; rm -rf "$TMP"

paso "4. Python: NumPy, puro y multiprocessing (réplicas y franjas)"
if [ "${RAPIDO:-0}" = "1" ]; then
  PY="$PY" NUMPY_RUNS=2 PURE_REPS=1 PURE_REGIONS="amazonia" NUMPY_REGIONS="amazonia" bash tools/benchmark_python_single.sh
else
  PY="$PY" NUMPY_RUNS=8 bash tools/benchmark_python_single.sh
fi
PY="$PY" REPS=$RUNS WORKERS="$THREADS" bash tools/benchmark_python_mp.sh $REG_ESCALA
PY="$PY" REPS=$RUNS WORKERS="$THREADS" ESTRATEGIA=franjas bash tools/benchmark_python_mp.sh $REG_FRANJAS

paso "5. Réplicas de referencia (Python puro) para la validación"
$PYTOOLS tools/pure_replicas.py --region amazonia --runs $PURE_A --jobs "$FIS"
[ "$PURE_P" -gt 0 ] && $PYTOOLS tools/pure_replicas.py --region pacifico --runs $PURE_P --jobs "$FIS"

regenerar_informe
