#!/usr/bin/env bash
# Tiempos de Python puro y NumPy (una simulación por ejecución, workers=1).
# Python puro solo se mide en tableros pequeños (guía §1.2); el resto se extrapola en el notebook.
# Uso (desde la raíz del repo):  bash tools/benchmark_python_single.sh
# Variables: NUMPY_RUNS=8  PURE_REPS=3  PURE_REGIONS="amazonia orinoquia"  NUMPY_REGIONS="amazonia orinoquia pacifico caribe andina"  PY=python3
# NumPy: 1 calentamiento + NUMPY_RUNS réplicas con semilla (sirven a la vez para los tiempos y para la validación).
set -euo pipefail
cd "$(dirname "$0")/../python"
NUMPY_RUNS=${NUMPY_RUNS:-8}; PURE_REPS=${PURE_REPS:-3}; PY=${PY:-python3}
PURE_REGIONS=${PURE_REGIONS:-"amazonia orinoquia"}
NUMPY_REGIONS=${NUMPY_REGIONS:-"amazonia orinoquia pacifico caribe andina"}
for region in $NUMPY_REGIONS; do
  $PY main.py numpy --region "$region" --runs "$NUMPY_RUNS" --warmup 1 --disable-animation 2>&1 | grep TIEMPO
done
for region in $PURE_REGIONS; do
  for rep in $(seq "$PURE_REPS"); do $PY main.py pure --region "$region" --disable-animation 2>&1 | grep TIEMPO; done
done
