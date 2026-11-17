import math
import numpy as np
import pandas as pd
from typing import Any

from domain.constants import AC_BITING_RATE, AC_CELL_POPULATION, AC_MAX_MOSQUITOES_PER_CELL, AC_MOSQUITO_MAX_AGE_DAYS, AC_MOSQUITO_WEATHER_DENSITY, \
    AC_NEIGHBOR_COUPLING, AC_MOSQUITO_NEIGHBORS_DENSITY, AC_STEPS_PER_MONTH, AC_STRAIN_PARAMS
from domain.enums import MalariaType


def ac_generate_initial_boards(
        xp: Any,
        df: Any,
):
    df_sorted = df.sort_values("fecha")
    df_by_date = df_sorted.groupby(["fecha"], sort=True)

    first_pop = float(df_by_date["poblacion_total"].sum().iloc[0])
    board_size = max(1, math.ceil(math.sqrt(first_pop / AC_CELL_POPULATION)))

    # Tomamos el promedio ponderado o media inicial de IPA (casos por cada 1.000 habitantes)
    IPA_falciparum = float(df_by_date["IPA_falciparum"].mean().iloc[0])
    IPA_vivax = float(df_by_date["IPA_vivax"].mean().iloc[0])

    p_init_falciparum = min(1.0, max(0.0, IPA_falciparum / float(AC_CELL_POPULATION)))
    p_init_vivax = min(1.0, max(0.0, IPA_vivax / float(AC_CELL_POPULATION)))

    # Humans - Falciparum (S + I + R = 1.000 por celda)
    humans_falciparum_infected_count_board = xp.random.binomial(
        n=AC_CELL_POPULATION,
        p=p_init_falciparum,
        size=(board_size, board_size),
    ).astype(xp.int32)
    humans_falciparum_recovered_count_board = xp.zeros((board_size, board_size), dtype=xp.int32)
    humans_falciparum_susceptible_count_board = (
        AC_CELL_POPULATION - humans_falciparum_infected_count_board
    ).astype(xp.int32)

    # Mosquitoes - Falciparum (Población basal inicial en equilibrio climático)
    init_mosq_total_f = xp.random.binomial(
        n=AC_MAX_MOSQUITOES_PER_CELL,
        p=0.25,
        size=(board_size, board_size),
    ).astype(xp.int32)
    mosquitoes_falciparum_infected_count_board = xp.random.binomial(
        n=init_mosq_total_f,
        p=min(1.0, p_init_falciparum * 2.0),
        size=(board_size, board_size),
    ).astype(xp.int32)
    mosquitoes_falciparum_susceptible_count_board = (
        init_mosq_total_f - mosquitoes_falciparum_infected_count_board
    ).astype(xp.int32)

    # Humans - Vivax (S + I + R = 1.000 por celda)
    humans_vivax_infected_count_board = xp.random.binomial(
        n=AC_CELL_POPULATION,
        p=p_init_vivax,
        size=(board_size, board_size),
    ).astype(xp.int32)
    humans_vivax_recovered_count_board = xp.zeros((board_size, board_size), dtype=xp.int32)
    humans_vivax_susceptible_count_board = (
        AC_CELL_POPULATION - humans_vivax_infected_count_board
    ).astype(xp.int32)

    # Mosquitoes - Vivax (Población basal inicial en equilibrio climático)
    init_mosq_total_v = xp.random.binomial(
        n=AC_MAX_MOSQUITOES_PER_CELL,
        p=0.25,
        size=(board_size, board_size),
    ).astype(xp.int32)
    mosquitoes_vivax_infected_count_board = xp.random.binomial(
        n=init_mosq_total_v,
        p=min(1.0, p_init_vivax * 2.0),
        size=(board_size, board_size),
    ).astype(xp.int32)
    mosquitoes_vivax_susceptible_count_board = (
        init_mosq_total_v - mosquitoes_vivax_infected_count_board
    ).astype(xp.int32)

    return {
        "board_size": board_size,
        "IPA_falciparum": IPA_falciparum,
        "IPA_vivax": IPA_vivax,
        "humans_falciparum_susceptible_count_board": humans_falciparum_susceptible_count_board,
        "humans_falciparum_infected_count_board": humans_falciparum_infected_count_board,
        "humans_falciparum_recovered_count_board": humans_falciparum_recovered_count_board,
        "mosquitoes_falciparum_susceptible_count_board": mosquitoes_falciparum_susceptible_count_board,
        "mosquitoes_falciparum_infected_count_board": mosquitoes_falciparum_infected_count_board,
        "humans_vivax_susceptible_count_board": humans_vivax_susceptible_count_board,
        "humans_vivax_infected_count_board": humans_vivax_infected_count_board,
        "humans_vivax_recovered_count_board": humans_vivax_recovered_count_board,
        "mosquitoes_vivax_susceptible_count_board": mosquitoes_vivax_susceptible_count_board,
        "mosquitoes_vivax_infected_count_board": mosquitoes_vivax_infected_count_board,
    }


def ac_count_neighbors(board: Any, xp: Any) -> Any:
    """
    Calcula la suma de los 8 vecinos de Moore en toda la cuadrícula 2D
    utilizando desplazamientos circulares (toroide) en GPU o CPU.

    Parameters:
        board: Tablero de conteos a analizar
        xp: Librería de NumPy o CuPy
    """

    int_board = board.astype(xp.int32)
    return (
        xp.roll(int_board,  1, axis=0) + xp.roll(int_board, -1, axis=0) +
        xp.roll(int_board,  1, axis=1) + xp.roll(int_board, -1, axis=1) +
        xp.roll(xp.roll(int_board,  1, axis=0),  1, axis=1) +
        xp.roll(xp.roll(int_board,  1, axis=0), -1, axis=1) +
        xp.roll(xp.roll(int_board, -1, axis=0),  1, axis=1) +
        xp.roll(xp.roll(int_board, -1, axis=0), -1, axis=1)
    )


def ac_to_numpy(arr: Any) -> np.ndarray:
    """Convierte un arreglo de CuPy o NumPy a np.ndarray en memoria de CPU."""

    if hasattr(arr, "get"):
        return arr.get()

    return np.asarray(arr)


_BINOMIAL_EXACTA = {"disponible": True}


def _binomial_aproximada(xp: Any, rng: Any, n: Any, p: Any) -> Any:
    """
    Respaldo aproximado, solo si el generador no ofrece binomial exacta vectorizada.
    Se elige la aproximación celda por celda (Poisson si mu < 10, Normal si no) y se usa la
    simetría Bin(n, p) = n - Bin(n, 1 - p) para trabajar siempre con q = min(p, 1 - p).
    """
    q = xp.minimum(p, 1.0 - p)
    mean = (n * q).astype(xp.float32)
    small = mean < 10.0
    poisson = rng.poisson(xp.where(small, xp.maximum(mean, 0.0), 0.0))
    std = xp.sqrt(xp.maximum(0.0, mean * (1.0 - q))).astype(xp.float32)
    normal = xp.rint(mean + std * rng.standard_normal(n.shape, dtype=xp.float32))
    x = xp.clip(xp.where(small, poisson, normal), 0, n)
    return xp.where(p > 0.5, n - x, x).astype(xp.int32)


def ac_fast_binomial(xp: Any, rng: Any, n: Any, p: Any) -> Any:
    """
    Muestreo Binomial(n, p) vectorizado sobre todo el tablero, en CPU (NumPy) o GPU (CuPy).

    Usa la binomial EXACTA del generador (numpy/cupy Generator.binomial, con n y p por celda).

    Por qué no la aproximación original (Poisson si max(n p) < 10, si no Normal para TODO el
    tablero): se comprobó que sesgaba el modelo cerca del umbral epidémico (en Pacífico, P.
    falciparum daba varias veces más casos que la referencia exacta de Python puro):
      1. decidir con el MÁXIMO del tablero aplicaba la Normal redondeada a celdas con media < 1
         (con media 0,5: P(x > 0) = 0,50 en vez de 0,39);
      2. la Poisson con p grande y n pequeño (recuperación, muerte del mosquito) subestima la
         media al recortar a n y casi duplica la varianza (n = 5, p = 0,53: 2,58 en vez de 2,65).
    Ver tools/test_fast_binomial.py. La exacta cuesta más por llamada, pero hace que NumPy, CuPy,
    Python puro y C++ muestreen la misma distribución, y la validación estadística sea limpia.
    """
    p = xp.clip(xp.broadcast_to(xp.asarray(p, dtype=xp.float64), n.shape), 0.0, 1.0)
    if _BINOMIAL_EXACTA["disponible"]:
        try:
            return rng.binomial(n, p).astype(xp.int32)
        except (AttributeError, NotImplementedError, TypeError, ValueError):
            _BINOMIAL_EXACTA["disponible"] = False
            import logging
            logging.getLogger("utils_npcp").warning(
                "El generador no soporta binomial exacta con arreglos: se usa la aproximación (ver docstring)")
    return _binomial_aproximada(xp, rng, n, p)


def ac_step_strain_day(
        xp: Any,
        rng: Any,
        strain: MalariaType,
        S_h: Any,
        I_h: Any,
        R_h: Any,
        S_m: Any,
        I_m: Any,
        temperature: float,
        humidity: float,
        monthly_precipitation: float,
        dt_days: float = 1.0,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """
    Ejecuta un paso temporal (dt_days) del modelo metapoblacional SIRS (Humanos) - SI (Mosquitos)
    para una cepa específica (P. falciparum o P. vivax).
    Retorna los tableros actualizados (S_h, I_h, R_h, S_m, I_m) y los nuevos casos del sub-paso.
    """

    params = AC_STRAIN_PARAMS[strain]

    # 1. Factores climáticos de la cepa
    if params["t_min"] < temperature < 38.0:
        f_t = math.exp(-((temperature - params["t_opt"]) ** 2) / (2.0 * (params["t_sigma"] ** 2)))
    else:
        f_t = 0.0

    g_h = 1.0 / (1.0 + math.exp(-0.15 * (humidity - 65.0)))
    h_p = min(1.0, max(0.05, monthly_precipitation / 200.0))
    climate_suitability = f_t * g_h * h_p

    # 2. Conteo promedio de los 8 vecinos de Moore
    I_h_neighbors_mean = ac_count_neighbors(I_h, xp) / 8.0
    I_m_neighbors_mean = ac_count_neighbors(I_m, xp) / 8.0
    M_alive = S_m + I_m
    M_alive_neighbors_mean = ac_count_neighbors(M_alive, xp) / 8.0

    # 3. Exposición efectiva (acoplando la celda actual con sus 8 vecinos)
    I_m_effective = ((1.0 - AC_NEIGHBOR_COUPLING) * I_m + AC_NEIGHBOR_COUPLING * I_m_neighbors_mean) / float(AC_CELL_POPULATION)
    I_h_effective = ((1.0 - AC_NEIGHBOR_COUPLING) * I_h + AC_NEIGHBOR_COUPLING * I_h_neighbors_mean) / float(AC_CELL_POPULATION)

    # 4. Dinámica Humana (S -> I -> R -> S + recaídas en P. vivax) escalada por dt_days
    rate_m_to_h = max(0.0, min(1.0, AC_BITING_RATE * params["prob_mosquito_to_human"]))
    p_infect_human = xp.clip(1.0 - xp.power(1.0 - rate_m_to_h, I_m_effective * dt_days), 0.0, 1.0)
    p_recover_h = min(1.0, dt_days / params["recuperation_days"])
    p_loss_imm_h = min(1.0, dt_days / params["immunity_loss_days"])

    new_infected_h = ac_fast_binomial(xp, rng, S_h, p_infect_human)
    new_recovered_h = ac_fast_binomial(xp, rng, I_h, p_recover_h)

    if params["relapse_rate"] > 0.0:
        p_relapse = min(1.0, 1.0 - (1.0 - params["relapse_rate"]) ** dt_days)
        relapsed_h = ac_fast_binomial(xp, rng, R_h, p_relapse)
    else:
        relapsed_h = xp.zeros_like(R_h)

    remaining_R_h = xp.maximum(0, R_h - relapsed_h)
    lost_immunity_h = ac_fast_binomial(xp, rng, remaining_R_h, p_loss_imm_h)

    S_h_next = S_h - new_infected_h + lost_immunity_h
    I_h_next = I_h + new_infected_h - new_recovered_h + relapsed_h
    R_h_next = remaining_R_h + new_recovered_h - lost_immunity_h

    # 5. Dinámica del Mosquito (Nacimiento -> S_m -> I_m -> Muerte) escalada por dt_days
    p_death_m = min(1.0, dt_days / float(AC_MOSQUITO_MAX_AGE_DAYS))
    dead_S_m = ac_fast_binomial(xp, rng, S_m, p_death_m)
    dead_I_m = ac_fast_binomial(xp, rng, I_m, p_death_m)

    S_m_surv = S_m - dead_S_m
    I_m_surv = I_m - dead_I_m

    empty_m_slots = xp.maximum(0, AC_MAX_MOSQUITOES_PER_CELL - (S_m_surv + I_m_surv))
    p_emergence = AC_MOSQUITO_WEATHER_DENSITY * climate_suitability * dt_days
    p_dispersion = AC_MOSQUITO_NEIGHBORS_DENSITY * (M_alive_neighbors_mean / float(AC_MAX_MOSQUITOES_PER_CELL)) * dt_days
    p_birth_m = xp.clip(p_emergence + p_dispersion, 0.0, 1.0)

    newborn_m = ac_fast_binomial(xp, rng, empty_m_slots, p_birth_m)

    rate_h_to_m = max(0.0, min(1.0, AC_BITING_RATE * params["prob_human_to_mosquito"] * f_t))
    p_infect_mosquito = xp.clip(1.0 - xp.power(1.0 - rate_h_to_m, I_h_effective * dt_days), 0.0, 1.0)
    new_infected_m = ac_fast_binomial(xp, rng, S_m_surv, p_infect_mosquito)

    S_m_next = S_m_surv - new_infected_m + newborn_m
    I_m_next = I_m_surv + new_infected_m

    step_new_cases = new_infected_h + relapsed_h
    return S_h_next, I_h_next, R_h_next, S_m_next, I_m_next, step_new_cases


def ac_monthly_schedule(df: Any) -> pd.DataFrame:
    """Agrega el DataFrame de una región por mes (igual que utils_pure)."""
    return (
        df.groupby("fecha", as_index=False)
        .agg({
            "temperatura_media_Celsius": "mean",
            "humedad_mediana_pct": "mean",
            "precipitacion_media_mm": "mean",
            "IPA_falciparum": "mean",
            "IPA_vivax": "mean",
            "falciparum": "sum",
            "vivax": "sum",
        })
        .sort_values("fecha")
        .reset_index(drop=True)
    )


BOARD_KEYS = (
    "humans_falciparum_susceptible_count_board",
    "humans_falciparum_infected_count_board",
    "humans_falciparum_recovered_count_board",
    "mosquitoes_falciparum_susceptible_count_board",
    "mosquitoes_falciparum_infected_count_board",
    "humans_vivax_susceptible_count_board",
    "humans_vivax_infected_count_board",
    "humans_vivax_recovered_count_board",
    "mosquitoes_vivax_susceptible_count_board",
    "mosquitoes_vivax_infected_count_board",
)


def ac_boards_to_device(xp: Any, ac_boards_data: dict) -> list:
    """Copia los 10 tableros (en el orden de BOARD_KEYS) a la memoria de xp (CPU -> GPU con CuPy)."""
    return [xp.asarray(ac_boards_data[k], dtype=xp.int32) for k in BOARD_KEYS]


def ac_simulate_on_device(xp: Any, df_monthly: pd.DataFrame, boards: list, seed: Any = None,
                          keep_snapshots: bool = False) -> dict:
    """
    Bucle de los 204 meses SIN transferencias a la CPU: los totales mensuales, los casos y
    (opcionalmente) las instantáneas se acumulan en arreglos del dispositivo. Así, con CuPy,
    el tiempo de cómputo y el de las copias GPU <-> CPU se pueden medir por separado (guía §6.4).

    Devuelve arreglos de xp: totals (meses x 10), cases (meses x 2) y snapshots (o None).
    """
    rng = xp.random.default_rng(seed)
    S_hf, I_hf, R_hf, S_mf, I_mf, S_hv, I_hv, R_hv, S_mv, I_mv = [b.copy() for b in boards]
    months = len(df_monthly)
    L = boards[0].shape[0]
    totals = xp.zeros((months, 10), dtype=xp.int64)
    cases = xp.zeros((months, 2), dtype=xp.int64)
    snapshots = xp.zeros((months, 10, L, L), dtype=xp.int32) if keep_snapshots else None

    for t, row in enumerate(df_monthly.itertuples()):
        dt_days = float(pd.Timestamp(row.fecha).days_in_month) / float(AC_STEPS_PER_MONTH)
        temp = float(row.temperatura_media_Celsius)
        hum = float(row.humedad_mediana_pct)
        precip = float(row.precipitacion_media_mm)

        for _ in range(AC_STEPS_PER_MONTH):
            S_hf, I_hf, R_hf, S_mf, I_mf, new_f = ac_step_strain_day(
                xp, rng, MalariaType.FALCIPARUM, S_hf, I_hf, R_hf, S_mf, I_mf, temp, hum, precip, dt_days
            )
            S_hv, I_hv, R_hv, S_mv, I_mv, new_v = ac_step_strain_day(
                xp, rng, MalariaType.VIVAX, S_hv, I_hv, R_hv, S_mv, I_mv, temp, hum, precip, dt_days
            )
            cases[t, 0] += new_f.sum(dtype=xp.int64)
            cases[t, 1] += new_v.sum(dtype=xp.int64)

        state = (S_hf, I_hf, R_hf, S_mf, I_mf, S_hv, I_hv, R_hv, S_mv, I_mv)
        totals[t] = xp.stack([b.sum(dtype=xp.int64) for b in state])
        if keep_snapshots:
            snapshots[t] = xp.stack(state)

    return {"totals": totals, "cases": cases, "snapshots": snapshots, "final": list(state)}


def ac_collect_results(xp: Any, df_monthly: pd.DataFrame, sim: dict, ac_boards_data: dict) -> pd.DataFrame:
    """
    Copia los resultados al host (GPU -> CPU con CuPy) y arma el historial con las mismas
    columnas que utils_pure. Rellena ac_boards_data con los tableros finales, el historial y
    las instantáneas (para la animación).
    """
    totals = ac_to_numpy(sim["totals"])
    cases = ac_to_numpy(sim["cases"])
    snapshots = ac_to_numpy(sim["snapshots"]) if sim["snapshots"] is not None else None
    for k, board in zip(BOARD_KEYS, sim["final"]):
        ac_boards_data[k] = ac_to_numpy(board)

    total_cells = float(ac_boards_data["board_size"] ** 2)
    history, frames = [], []
    for t, row in enumerate(df_monthly.itertuples()):
        tt = totals[t]
        cf, cv = int(cases[t, 0]), int(cases[t, 1])
        history.append({
            "fecha": row.fecha,
            "falciparum_S_h": int(tt[0]), "falciparum_I_h": int(tt[1]), "falciparum_R_h": int(tt[2]),
            "falciparum_S_m": int(tt[3]), "falciparum_I_m": int(tt[4]),
            "falciparum_nuevos_casos_mes": cf,
            "IPA_falciparum_simulado": cf / total_cells,
            "IPA_falciparum_real": float(row.IPA_falciparum),
            "vivax_S_h": int(tt[5]), "vivax_I_h": int(tt[6]), "vivax_R_h": int(tt[7]),
            "vivax_S_m": int(tt[8]), "vivax_I_m": int(tt[9]),
            "vivax_nuevos_casos_mes": cv,
            "IPA_vivax_simulado": cv / total_cells,
            "IPA_vivax_real": float(row.IPA_vivax),
        })
        if snapshots is not None:
            frames.append({"fecha": pd.Timestamp(row.fecha), **{k: snapshots[t, i] for i, k in enumerate(BOARD_KEYS)}})

    df_sim = pd.DataFrame(history)
    ac_boards_data["simulation_history"] = df_sim
    ac_boards_data["frames_snapshots"] = frames
    return df_sim


def ac_apply_evolution_rules(
        xp: Any,
        df: Any,
        ac_boards_data: Any,
        seed: Any = None,
        keep_snapshots: bool = True,
) -> pd.DataFrame:
    """
    Evoluciona los tableros de P. falciparum y P. vivax a lo largo de las fechas del DataFrame,
    guardando el historial numérico y (si keep_snapshots) las instantáneas 2D para la animación.
    Equivale a: copiar tableros al dispositivo -> simular -> traer resultados.
    """
    df_monthly = ac_monthly_schedule(df)
    boards = ac_boards_to_device(xp, ac_boards_data)
    sim = ac_simulate_on_device(xp, df_monthly, boards, seed=seed, keep_snapshots=keep_snapshots)
    return ac_collect_results(xp, df_monthly, sim, ac_boards_data)
