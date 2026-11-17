"""
Prueba del kernel CUDA (python/adapters/kernel_automata.cu) SIN GPU.

Compila el mismo archivo .cu con g++ (-DEMULAR_CPU), lo llama con ctypes celda por celda y
ejecuta la simulación completa con la misma lógica de host que usa la GPU
(utils_cuda.simular). Luego valida los resultados como cualquier otra implementación:
  1. invariantes (conservación de población, cota de mosquitos) en cada mes;
  2. comparación estadística contra las réplicas de Python puro (tools/compare_stats.py).

Uso (desde la raíz del repo):
    python tools/test_kernel_cpu.py --region amazonia --runs 8
Escribe en outputs/kernel_cpu_test/ (no toca outputs/times.csv).
"""
import argparse
import ctypes
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
from adapters.utils_cuda import KERNEL_PATH, simular  # noqa: E402
from adapters.utils_npcp import BOARD_KEYS, ac_collect_results, ac_generate_initial_boards, ac_monthly_schedule  # noqa: E402

NOMBRES = {"amazonia": "Amazonia", "andina": "Andina", "caribe": "Caribe", "orinoquia": "Orinoquia", "pacifico": "Pacifico"}


def compilar() -> ctypes.CDLL:
    lib = Path(tempfile.mkdtemp()) / "kernel_cpu.so"
    subprocess.run(["g++", "-O2", "-x", "c++", "-shared", "-fPIC", "-DEMULAR_CPU", str(KERNEL_PATH), "-o", str(lib)],
                   check=True)
    k = ctypes.CDLL(str(lib))
    P = ctypes.POINTER(ctypes.c_int)
    k.lanzar_cpu.argtypes = [ctypes.c_int] + [P] * 11 + [ctypes.c_int] + [ctypes.c_double] * 5 + [ctypes.c_int] + \
                            [ctypes.c_double] * 3 + [ctypes.c_ulonglong]
    return k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="amazonia", choices=list(NOMBRES))
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=777)
    a = ap.parse_args()

    k = compilar()
    ptr = lambda arr: arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))  # noqa: E731

    def lanzar(cur, nxt, casos, L, c, fl):
        for arr in (*cur, *nxt, casos):
            assert arr.flags["C_CONTIGUOUS"] and arr.dtype == np.int32
        k.lanzar_cpu(L * L, *[ptr(x) for x in cur], *[ptr(x) for x in nxt], ptr(casos), L, *c[:5], c[5], *c[6:], fl)

    df = pd.read_parquet(ROOT / "data" / "GAM_regional_enriquecido.parquet")
    df = df[df.region == NOMBRES[a.region]]
    df_monthly = ac_monthly_schedule(df)
    out = ROOT / "outputs" / "kernel_cpu_test"
    shutil.rmtree(out, ignore_errors=True)
    (out / "metrics").mkdir(parents=True)
    for run in range(a.runs):
        np.random.seed(a.seed + run)
        data = {"df": df, **ac_generate_initial_boards(np, df)}
        boards = [np.ascontiguousarray(data[key], dtype=np.int32) for key in BOARD_KEYS]
        sim = simular(np, lanzar, df_monthly, boards, a.seed + run)
        hist = ac_collect_results(np, df_monthly, sim, data).copy()
        hist.insert(0, "region", a.region)
        hist.insert(0, "paradigm", "cuda")
        hist.to_csv(out / "metrics" / f"cuda_{a.region}_run{run}.csv", index=False)
        print(f"réplica {run}: casos falciparum {hist.falciparum_nuevos_casos_mes.sum()}, vivax {hist.vivax_nuevos_casos_mes.sum()}",
              flush=True)

    r1 = subprocess.run([sys.executable, str(ROOT / "tools" / "validate_metrics.py"), *map(str, (out / "metrics").glob("*.csv"))])
    for f in (ROOT / "outputs" / "metrics").glob(f"pure_{a.region}_run*.csv"):
        shutil.copy(f, out / "metrics")
    r2 = subprocess.run([sys.executable, str(ROOT / "tools" / "compare_stats.py"), "--metrics-dir", str(out / "metrics")])
    return r1.returncode or r2.returncode


if __name__ == "__main__":
    sys.exit(main())
