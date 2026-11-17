import argparse
import csv
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import random
import time
from typing import Any, Optional

from adapters.utils_animations import ac_animate_boards
from adapters.utils_npcp import (
    ac_boards_to_device,
    ac_collect_results,
    ac_generate_initial_boards,
    ac_monthly_schedule,
    ac_simulate_on_device,
    ac_to_numpy,
)
from adapters.utils_mp import ac_apply_evolution_rules as mp_apply_evolution_rules
from adapters.utils_parallel import DEFAULT_RUNS, DEFAULT_SEED, run_parallel_replicas
from adapters.utils_pure import (
    ac_apply_evolution_rules as pure_apply_evolution_rules,
    ac_generate_initial_boards as pure_generate_initial_boards,
)


# Increase the limit to 60 MB
plt.rcParams["animation.embed_limit"] = 100.0

# Configuración básica para mostrar mensajes en consola
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
)

logger = logging.getLogger("main")

# Carpeta de resultados. Por defecto ../outputs; en Google Colab (GPU) se usa ../outputs/gpu
# con la variable de entorno HPC_OUTPUTS para no mezclar los resultados de dos equipos.
OUT = Path(os.environ.get("HPC_OUTPUTS", "../outputs"))
for _sub in ("metrics", "animations"):
    (OUT / _sub).mkdir(parents=True, exist_ok=True)

# Datasets

df_by_regions = pd.read_parquet("../data/GAM_regional_enriquecido.parquet")
logger.debug(f"[DF] Tamaño del DataFrame (Por regiones): {len(df_by_regions)} datos")

# Handlers

def handle_pure(args: argparse.Namespace):
    logger.debug("=== Código ejecutándose en Python puro secuencial ===")
    selected_region = args.region if args.region.lower() != "todo" else None

    ac_data_store_by_regions = {}
    for region, new_df in df_by_regions.groupby("region"):
        if selected_region is not None and region.lower() != selected_region.lower():
            continue

        ac_data_store_by_regions[region] = {
            "df": new_df,
            **pure_generate_initial_boards(new_df),
        }

    for region, data in ac_data_store_by_regions.items():
        logger.info("[CELDAS] {0} celdas a analizar para la región {1}".format(
            (data["board_size"] ** 2) * 10 * 204,
            region,
        ))
    
    for region, data in ac_data_store_by_regions.items():
        start = time.perf_counter()
        pure_apply_evolution_rules(data["df"], data)
        elapsed = time.perf_counter() - start

        with open(f"{OUT}/times.csv", "a") as times_csv_file:
            csv_writer = csv.writer(times_csv_file)
            csv_writer.writerow([
                time.time(), # execution_timestamp
                "python", # language
                "pure", # paradigm
                region.lower(), # region
                1, # workers
                data["board_size"], # board_size
                204, # iterations
                elapsed, # elapsed_time
            ])

        logger.info(f"[TIEMPO] Tiempo de ejecución para {region}: {elapsed:.2f} s")

        # Guardar series temporales para análisis y validación estadística
        metrics_dir = OUT / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        if "simulation_history" in data:
            metrics_df = data["simulation_history"].copy()
            metrics_df.insert(0, "region", region.lower())
            metrics_df.insert(0, "paradigm", "pure")
            metrics_df.to_csv(metrics_dir / f"pure_{region.lower()}.csv", index=False)

        if not args.disable_animation:
            with open(f"{OUT}/animations/pure_{region.lower()}.html", "w+") as file:
                file.write(ac_animate_boards(region, data))


def _append_times(paradigm: str, region: str, workers: int, board_size: int, elapsed: float) -> None:
    """Añade una fila a times.csv (formato de AGENTS.md §8)."""
    with open(OUT / "times.csv", "a") as times_csv_file:
        csv.writer(times_csv_file).writerow([
            time.time(), "python", paradigm, region.lower(), workers, board_size, 204, elapsed,
        ])


def _append_times_gpu(paradigm: str, region: str, board_size: int, t_h2d: float, t_compute: float, t_d2h: float) -> None:
    """Desglose de tiempos en GPU (guía §6.4 y gráfica G7): copia a la GPU, cómputo y copia de vuelta."""
    path = OUT / "times_gpu.csv"
    nuevo = not path.exists()
    with open(path, "a") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["execution_timestamp", "paradigm", "region", "board_size", "t_h2d", "t_compute", "t_d2h", "t_total"])
        w.writerow([time.time(), paradigm, region.lower(), board_size, t_h2d, t_compute, t_d2h, t_h2d + t_compute + t_d2h])


def _save_metrics(data: dict, paradigm: str, region: str, run: int, runs: int) -> None:
    metrics_df = data["simulation_history"].copy()
    metrics_df.insert(0, "region", region.lower())
    metrics_df.insert(0, "paradigm", paradigm)
    sufijo = f"_run{run}" if runs > 1 else ""
    metrics_df.to_csv(OUT / "metrics" / f"{paradigm}_{region.lower()}{sufijo}.csv", index=False)


def _run_npcp(xp: Any, paradigm_name: str, args: argparse.Namespace) -> None:
    """
    Ejecuta la versión vectorizada (NumPy en CPU o CuPy en GPU), --runs réplicas por región.

    Cada réplica se mide en tres partes (guía §6.4):
      1. copia de los tableros iniciales a la memoria del dispositivo (CPU -> GPU),
      2. cómputo de los 204 meses (sin transferencias intermedias),
      3. copia de los resultados de vuelta (GPU -> CPU).
    En CuPy se miden con eventos de CUDA y sincronización. elapsed_time de times.csv es SOLO el
    cómputo (como en las demás versiones); las tres partes van a times_gpu.csv (solo GPU).
    Antes de medir se hace una ejecución de calentamiento (en CuPy compila los kernels).
    """
    es_gpu = xp.__name__ != "numpy"
    selected_region = args.region if args.region.lower() != "todo" else None

    def reloj():
        if es_gpu:
            ev = xp.cuda.Event()
            ev.record()
            return ev
        return time.perf_counter()

    def segundos(a, b):
        if es_gpu:
            b.synchronize()
            return xp.cuda.get_elapsed_time(a, b) / 1000.0
        return b - a

    for region, new_df in df_by_regions.groupby("region"):
        if selected_region is not None and region.lower() != selected_region.lower():
            continue
        df_monthly = ac_monthly_schedule(new_df)

        for run in range(-args.warmup, args.runs):  # las réplicas negativas son de calentamiento
            seed = args.seed + max(run, 0) + (1_000_000 if run < 0 else 0)
            np.random.seed(seed)  # tableros iniciales en el host (CPU), igual para NumPy y CuPy
            data = {"df": new_df, **ac_generate_initial_boards(np, new_df)}
            keep = run == 0 and not args.disable_animation
            if run == 0:
                logger.info(f"[CELDAS] {(data['board_size'] ** 2) * 10 * 204} celdas a analizar para la región {region}")

            t0 = reloj()
            boards = ac_boards_to_device(xp, data)
            t1 = reloj()
            sim = ac_simulate_on_device(xp, df_monthly, boards, seed=seed, keep_snapshots=keep)
            t2 = reloj()
            sim = {k: (ac_to_numpy(v) if v is not None and k != "final" else v) for k, v in sim.items()}
            sim["final"] = [ac_to_numpy(b) for b in sim["final"]]
            t3 = reloj()
            t_h2d, t_compute, t_d2h = segundos(t0, t1), segundos(t1, t2), segundos(t2, t3)
            if run < 0:
                continue

            # Todo lo siguiente ocurre después de detener el cronómetro
            ac_collect_results(np, df_monthly, sim, data)
            _append_times(paradigm_name, region, 1, data["board_size"], t_compute)
            if es_gpu:
                _append_times_gpu(paradigm_name, region, data["board_size"], t_h2d, t_compute, t_d2h)
            _save_metrics(data, paradigm_name, region, run, args.runs)
            logger.info(
                f"[TIEMPO] {paradigm_name} {region} réplica {run}: cómputo {t_compute:.3f} s"
                + (f" | copia a GPU {t_h2d * 1000:.1f} ms, copia de vuelta {t_d2h * 1000:.1f} ms" if es_gpu else "")
            )
            if keep:
                with open(OUT / "animations" / f"{paradigm_name}_{region.lower()}.html", "w+") as file:
                    file.write(ac_animate_boards(region, data))


def handle_numpy(args: argparse.Namespace):
    logger.debug("=== Código ejecutándose con NumPy ===")
    _run_npcp(np, "numpy", args)


def _cupy_disponible():
    """Devuelve el módulo cupy si hay una GPU NVIDIA utilizable; si no, None."""
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() == 0:
            return None
        return cp
    except Exception:
        return None


def handle_cupy(args: argparse.Namespace):
    cp = _cupy_disponible()
    if cp is None:
        logger.error("CuPy no está instalado o no hay GPU NVIDIA. Ejecute esta versión en Google Colab (ver colab/).")
        return
    logger.debug("=== Código ejecutándose con CuPy ===")
    _run_npcp(cp, "cupy", args)


def handle_cuda(args: argparse.Namespace):
    cp = _cupy_disponible()
    if cp is None:
        logger.error("El kernel CUDA necesita CuPy y una GPU NVIDIA. Ejecútelo en Google Colab (ver colab/).")
        return
    from adapters.utils_cuda import run_cuda
    run_cuda(cp, df_by_regions, args, _append_times, _append_times_gpu, _save_metrics, logger)


def handle_multiprocessing(args: argparse.Namespace):
    """
    Paralelo en CPU con multiprocessing. Dos estrategias:
      - replicas (por defecto): R réplicas Monte Carlo repartidas entre p procesos
        (adapters/utils_parallel.py; "nivel natural" según la guía, §5.6 paso 5).
      - franjas: una simulación repartida por franjas de filas entre p procesos con
        memoria compartida (adapters/utils_mp.py).
    """
    if args.estrategia == "franjas":
        handle_multiprocessing_franjas(args)
        return
    if args.runs is None:
        args.runs = DEFAULT_RUNS

    logger.debug("=== Código ejecutándose con multiprocessing (réplicas) ===")
    selected_region = args.region if args.region.lower() != "todo" else None

    for region, new_df in df_by_regions.groupby("region"):
        if selected_region is not None and region.lower() != selected_region.lower():
            continue

        logger.info(
            f"[MULTIPROCESSING] Región {region}: {args.runs} réplicas, "
            f"{args.workers} procesos, kernel {args.kernel}"
        )
        elapsed, results = run_parallel_replicas(
            new_df,
            workers=args.workers,
            runs=args.runs,
            seed=args.seed,
            kernel=args.kernel,
            keep_first_data=not args.disable_animation,
        )

        # Todo lo siguiente ocurre después de detener el cronómetro
        with open(f"{OUT}/times.csv", "a") as times_csv_file:
            csv_writer = csv.writer(times_csv_file)
            csv_writer.writerow([
                time.time(), # execution_timestamp
                "python", # language
                "multiprocessing", # paradigm
                region.lower(), # region
                args.workers, # workers (procesos activos, exacto)
                results[0]["board_size"], # board_size
                204, # iterations
                elapsed, # elapsed_time (lote completo de réplicas)
            ])

        logger.info(
            f"[TIEMPO] Tiempo de ejecución para {region}: {elapsed:.2f} s "
            f"({args.runs} réplicas, {args.workers} procesos)"
        )

        metrics_dir = OUT / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        for result in results:
            metrics_df = result["simulation_history"].copy()
            metrics_df.insert(0, "region", region.lower())
            metrics_df.insert(0, "paradigm", "multiprocessing")
            metrics_df.to_csv(
                metrics_dir / f"multiprocessing_{region.lower()}_run{result['run_id']}.csv",
                index=False,
            )

        if not args.disable_animation:
            with open(f"{OUT}/animations/multiprocessing_{region.lower()}.html", "w+") as file:
                file.write(ac_animate_boards(region, results[0]["data"]))


def handle_multiprocessing_franjas(args: argparse.Namespace):
    """
    Descomposición de dominio: el tablero se reparte por franjas de filas entre p procesos
    persistentes que comparten memoria (SharedMemory) y se sincronizan con barreras.
    Cada réplica (--runs) es una simulación completa con semilla propia.
    """
    logger.debug("=== Código ejecutándose con multiprocessing (franjas de filas) ===")
    selected_region = args.region if args.region.lower() != "todo" else None
    requested_workers = args.workers if args.workers is not None else (os.cpu_count() or 1)
    runs = args.runs if args.runs is not None else 1
    paradigm = "multiprocessing_franjas"

    for region, new_df in df_by_regions.groupby("region"):
        if selected_region is not None and region.lower() != selected_region.lower():
            continue

        for run in range(runs):
            # Semillas: inicialización con seed + run; el worker k usa (seed + 1000 * run) + k,
            # de modo que réplicas distintas nunca comparten flujo aleatorio.
            init_rng = random.Random(args.seed + run)
            data = {"df": new_df, **pure_generate_initial_boards(new_df, rng=init_rng)}
            if run == 0:
                logger.info("[CELDAS] {0} celdas a analizar para la región {1}".format(
                    (data["board_size"] ** 2) * 10 * 204, region,
                ))

            mp_apply_evolution_rules(new_df, data, workers=requested_workers, seed=args.seed + 1000 * run)
            workers = data["workers"]  # Procesos efectivos (acotados al tamaño del tablero)
            elapsed = data["compute_time"]  # Excluye el arranque de procesos, I/O y renderizado

            # Todo lo siguiente ocurre después de detener el cronómetro
            with open(f"{OUT}/times.csv", "a") as times_csv_file:
                csv_writer = csv.writer(times_csv_file)
                csv_writer.writerow([
                    time.time(), # execution_timestamp
                    "python", # language
                    paradigm, # paradigm
                    region.lower(), # region
                    workers, # workers (procesos efectivos, exacto)
                    data["board_size"], # board_size
                    204, # iterations
                    elapsed, # elapsed_time
                ])

            logger.info(
                f"[TIEMPO] Tiempo de ejecución para {region} con {workers} procesos (réplica {run}): {elapsed:.2f} s "
                f"(arranque de procesos: {data['setup_time']:.2f} s, no incluido)"
            )

            metrics_dir = OUT / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            metrics_df = data["simulation_history"].copy()
            metrics_df.insert(0, "region", region.lower())
            metrics_df.insert(0, "paradigm", paradigm)
            sufijo = f"_run{run}" if runs > 1 else ""
            metrics_df.to_csv(metrics_dir / f"{paradigm}_{region.lower()}{sufijo}.csv", index=False)

            if run == 0 and not args.disable_animation:
                with open(f"{OUT}/animations/{paradigm}_{region.lower()}.html", "w+") as file:
                    file.write(ac_animate_boards(region, data))


def main():
    times_csv_file = OUT / "times.csv"
    if not times_csv_file.exists():
        with open(f"{OUT}/times.csv", "w+") as times_csv_file:
            csv_writer = csv.writer(times_csv_file)
            csv_writer.writerow([
                "execution_timestamp",
                "language",
                "paradigm",
                "region",
                "workers",
                "board_size",
                "iterations",
                "elapsed_time"
            ])

    parser = argparse.ArgumentParser(description="CLI para el proyecto final de HPC")
    subparsers = parser.add_subparsers(
        dest="command",
        help="Comandos disponibles",
        required=True,
    )

    # Subcommand: Pure
    parser_pure = subparsers.add_parser(
        help="Ejecutar código puro (Secuencial)",
        name="pure",
    )
    parser_pure.add_argument(
        "--disable-animation",
        action=argparse.BooleanOptionalAction,
        help="No generar animación HTML",
        required=False,
    )
    parser_pure.add_argument(
        "--region",
        choices=["amazonia", "andina", "caribe", "orinoquia", "pacifico", "todo"],
        default="Pacifico",
        help="Región seleccionada para ejecutar el modelo, por defecto: Pacifico",
        required=False,
        type=str,
    )
    parser_pure.set_defaults(func=handle_pure)

    # Subcommands: NumPy, CuPy y kernel CUDA (mismos argumentos)
    for nombre, ayuda, funcion in (
        ("numpy", "Ejecutar código vectorizado con NumPy (CPU)", handle_numpy),
        ("cupy", "Ejecutar código vectorizado con CuPy (GPU NVIDIA; ver colab/)", handle_cupy),
        ("cuda", "Ejecutar el kernel CUDA escrito a mano con cupy.RawKernel (GPU NVIDIA; ver colab/)", handle_cuda),
    ):
        sub = subparsers.add_parser(help=ayuda, name=nombre)
        sub.add_argument(
            "--disable-animation",
            action=argparse.BooleanOptionalAction,
            help="No generar animación HTML",
            required=False,
        )
        sub.add_argument(
            "--region",
            choices=["amazonia", "andina", "caribe", "orinoquia", "pacifico", "todo"],
            default="Pacifico",
            help="Región seleccionada para ejecutar el modelo, por defecto: Pacifico",
            required=False,
            type=str,
        )
        sub.add_argument("--runs", default=1, type=int, required=False,
                         help="Réplicas a medir (una fila de times.csv por réplica), por defecto: 1")
        sub.add_argument("--seed", default=12345, type=int, required=False,
                         help="Semilla base (la réplica k usa seed + k), por defecto: 12345")
        sub.add_argument("--warmup", default=1, type=int, required=False,
                         help="Ejecuciones de calentamiento no registradas (en GPU compilan los kernels), por defecto: 1")
        sub.set_defaults(func=funcion)


    # Subcommand: multiprocessing (paralelo CPU)
    parser_mp = subparsers.add_parser(
        help="Ejecutar réplicas Monte Carlo en paralelo con multiprocessing",
        name="multiprocessing",
    )
    parser_mp.add_argument(
        "--disable-animation",
        action=argparse.BooleanOptionalAction,
        help="No generar animación HTML",
        required=False,
    )
    parser_mp.add_argument(
        "--region",
        choices=["amazonia", "andina", "caribe", "orinoquia", "pacifico", "todo"],
        default="Pacifico",
        help="Región seleccionada para ejecutar el modelo, por defecto: Pacifico",
        required=False,
        type=str,
    )
    parser_mp.add_argument(
        "--estrategia",
        choices=["replicas", "franjas"],
        default="replicas",
        help="replicas: R simulaciones independientes en paralelo (por defecto); "
             "franjas: una simulación repartida por filas con memoria compartida",
        required=False,
        type=str,
    )
    parser_mp.add_argument(
        "--workers",
        default=4,
        help="Cantidad de procesos de trabajo (p), por defecto: 4",
        required=False,
        type=int,
    )
    parser_mp.add_argument(
        "--runs",
        default=None,
        help=f"Cantidad de réplicas Monte Carlo (R), por defecto: {DEFAULT_RUNS} en replicas y 1 en franjas",
        required=False,
        type=int,
    )
    parser_mp.add_argument(
        "--seed",
        default=DEFAULT_SEED,
        help=f"Semilla base (la réplica k usa seed + k), por defecto: {DEFAULT_SEED}",
        required=False,
        type=int,
    )
    parser_mp.add_argument(
        "--kernel",
        choices=["numpy", "pure"],
        default="numpy",
        help="Implementación de cada réplica (solo estrategia replicas): numpy (por defecto) o pure",
        required=False,
        type=str,
    )
    parser_mp.set_defaults(func=handle_multiprocessing)

    # Initialize parser
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
