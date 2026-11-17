"""
Versión con kernel CUDA escrito a mano (paradigm="cuda"), ejecutado desde Python con
cupy.RawKernel (la guía lo acepta: §1.2 ítem 8 y §6.2, "05_cupy_rawkernel.py").

Diferencia con la versión CuPy: CuPy encadena muchas operaciones de arreglo (roll, power,
clip, binomial...), y cada una recorre el tablero completo en memoria. Aquí un solo kernel
hace TODO el sub-paso de una cepa: un hilo por celda lee sus 8 vecinas, sortea los 8 flujos
binomiales y escribe la celda del tablero siguiente (doble buffer). Así la memoria se recorre
una sola vez por sub-paso.

El kernel está en kernel_automata.cu. La lógica del host está separada del lanzamiento del
kernel para poder probarla en CPU sin GPU (tools/test_kernel_cpu.py compila el mismo .cu con
g++ y lo llama con ctypes).
"""
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from adapters.utils_npcp import (
    BOARD_KEYS,
    ac_collect_results,
    ac_generate_initial_boards,
    ac_monthly_schedule,
    ac_to_numpy,
)
from adapters.utils_pure import calc_climate_suitability
from domain.constants import AC_BITING_RATE, AC_MOSQUITO_MAX_AGE_DAYS, AC_STEPS_PER_MONTH, AC_STRAIN_PARAMS
from domain.enums import MalariaType

KERNEL_PATH = Path(__file__).with_name("kernel_automata.cu")
HILOS_POR_BLOQUE = 256
CEPAS = (MalariaType.FALCIPARUM, MalariaType.VIVAX)
_MASK = (1 << 64) - 1


def _splitmix(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK
    return x ^ (x >> 31)


def flujo(seed: int, mes: int, sub: int, cepa: int) -> int:
    """Semilla del sub-paso: mezcla de (semilla, mes, sub-paso, cepa)."""
    h = _splitmix(seed & _MASK)
    for v in (mes, sub, cepa):
        h = _splitmix(h ^ v)
    return h


def constantes(strain: MalariaType, T: float, H: float, P: float, dt: float) -> tuple:
    """Probabilidades del sub-paso (iguales a utils_pure.ac_step_strain_day)."""
    prm = AC_STRAIN_PARAMS[strain]
    idoneidad, f_t = calc_climate_suitability(T, H, P, prm["t_min"], prm["t_opt"], prm["t_sigma"])
    tasa_m_h = max(0.0, min(1.0, AC_BITING_RATE * prm["prob_mosquito_to_human"]))
    tasa_h_m = max(0.0, min(1.0, AC_BITING_RATE * prm["prob_human_to_mosquito"] * f_t))
    p_rec = min(1.0, dt / prm["recuperation_days"])
    p_perdida = min(1.0, dt / prm["immunity_loss_days"])
    rho = prm["relapse_rate"]
    p_recaida = min(1.0, 1.0 - (1.0 - rho) ** dt) if rho > 0 else 0.0
    p_muerte = min(1.0, dt / float(AC_MOSQUITO_MAX_AGE_DAYS))
    return tasa_m_h, tasa_h_m, p_rec, p_perdida, p_recaida, int(rho > 0), p_muerte, idoneidad, dt


def simular(xp: Any, lanzar: Callable, df_monthly, boards: list, seed: int) -> dict:
    """
    Bucle de los 204 meses con el kernel. `boards` son 10 arreglos de xp (orden BOARD_KEYS).
    `lanzar(cur, nxt, casos, L, consts, flujo)` ejecuta el kernel sobre todas las celdas.
    Devuelve el mismo diccionario que utils_npcp.ac_simulate_on_device.
    """
    L = boards[0].shape[0]
    cur = [[b.copy() for b in boards[0:5]], [b.copy() for b in boards[5:10]]]
    nxt = [[xp.empty_like(b) for b in grupo] for grupo in cur]
    casos = xp.zeros((L, L), dtype=xp.int32)
    meses = len(df_monthly)
    totals = xp.zeros((meses, 10), dtype=xp.int64)
    cases = xp.zeros((meses, 2), dtype=xp.int64)

    for t, row in enumerate(df_monthly.itertuples()):
        dt = float(pd.Timestamp(row.fecha).days_in_month) / AC_STEPS_PER_MONTH
        T, H, P = float(row.temperatura_media_Celsius), float(row.humedad_mediana_pct), float(row.precipitacion_media_mm)
        k = [constantes(s, T, H, P, dt) for s in CEPAS]
        for sub in range(AC_STEPS_PER_MONTH):
            for c in range(2):
                lanzar(cur[c], nxt[c], casos, L, k[c], flujo(seed, t, sub, c))
                cases[t, c] += casos.sum(dtype=xp.int64)
                cur[c], nxt[c] = nxt[c], cur[c]
        estado = cur[0] + cur[1]
        totals[t] = xp.stack([b.sum(dtype=xp.int64) for b in estado])

    return {"totals": totals, "cases": cases, "snapshots": None, "final": cur[0] + cur[1]}


def lanzador_cupy(cp: Any) -> Callable:
    """Compila el kernel con NVRTC (cupy.RawKernel) y devuelve la función de lanzamiento."""
    kernel = cp.RawKernel(KERNEL_PATH.read_text(encoding="utf-8"), "paso_cepa")

    def lanzar(cur, nxt, casos, L, k, fl):
        n = L * L
        bloques = (n + HILOS_POR_BLOQUE - 1) // HILOS_POR_BLOQUE
        tasa_m_h, tasa_h_m, p_rec, p_perdida, p_recaida, hay, p_muerte, idon, dt = k
        kernel((bloques,), (HILOS_POR_BLOQUE,), (
            *cur, *nxt, casos, np.int32(L),
            np.float64(tasa_m_h), np.float64(tasa_h_m), np.float64(p_rec), np.float64(p_perdida),
            np.float64(p_recaida), np.int32(hay), np.float64(p_muerte), np.float64(idon), np.float64(dt),
            np.uint64(fl),
        ))
    return lanzar


def run_cuda(cp, df_by_regions, args, append_times, append_times_gpu, save_metrics, logger) -> None:
    """Comando `main.py cuda`: mismas salidas y misma medición que `main.py cupy`."""
    lanzar = lanzador_cupy(cp)
    selected = args.region if args.region.lower() != "todo" else None

    def reloj():
        ev = cp.cuda.Event()
        ev.record()
        return ev

    def seg(a, b):
        b.synchronize()
        return cp.cuda.get_elapsed_time(a, b) / 1000.0

    for region, new_df in df_by_regions.groupby("region"):
        if selected is not None and region.lower() != selected.lower():
            continue
        df_monthly = ac_monthly_schedule(new_df)
        for run in range(-args.warmup, args.runs):
            seed = args.seed + max(run, 0) + (1_000_000 if run < 0 else 0)
            np.random.seed(seed)
            data = {"df": new_df, **ac_generate_initial_boards(np, new_df)}
            t0 = reloj()
            boards = [cp.asarray(data[k], dtype=cp.int32) for k in BOARD_KEYS]
            t1 = reloj()
            sim = simular(cp, lanzar, df_monthly, boards, seed)
            t2 = reloj()
            sim = {"totals": ac_to_numpy(sim["totals"]), "cases": ac_to_numpy(sim["cases"]), "snapshots": None,
                   "final": [ac_to_numpy(b) for b in sim["final"]]}
            t3 = reloj()
            t_h2d, t_comp, t_d2h = seg(t0, t1), seg(t1, t2), seg(t2, t3)
            if run < 0:
                continue
            ac_collect_results(np, df_monthly, sim, data)
            append_times("cuda", region, 1, data["board_size"], t_comp)
            append_times_gpu("cuda", region, data["board_size"], t_h2d, t_comp, t_d2h)
            save_metrics(data, "cuda", region, run, args.runs)
            logger.info(f"[TIEMPO] cuda {region} réplica {run}: cómputo {t_comp:.3f} s | copia a GPU "
                        f"{t_h2d * 1000:.1f} ms, copia de vuelta {t_d2h * 1000:.1f} ms")
