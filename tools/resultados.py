"""
Lee las tablas que genera notebooks/analisis.ipynb (outputs/figures/tabla_*.csv) y
devuelve un diccionario de valores ya formateados para el informe y la presentación.

Así NINGÚN número se escribe a mano: tras volver a medir en otro equipo basta con
ejecutar el notebook y luego tools/build_informe.py y tools/build_presentacion.py.

Uso:  from resultados import cargar;  R = cargar()
      python tools/resultados.py        # imprime las claves y valores (para revisar)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"

NOMBRE = {"pure": "Python puro", "numpy": "NumPy", "multiprocessing": "multiprocessing", "multiprocessing_franjas": "multiprocessing (franjas)", "cupy": "CuPy",
          "serial": "C++ secuencial", "openmp": "OpenMP", "mpi": "MPI", "cuda": "CUDA"}
REGION = {"amazonia": "Amazonía", "orinoquia": "Orinoquía", "pacifico": "Pacífico", "caribe": "Caribe", "andina": "Andina"}
PEND = "[PENDIENTE: ejecutar el notebook]"


def n(x, d=2):
    """Número con coma decimal (estilo español)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


def tabla_md(df: pd.DataFrame, align=None) -> str:
    cols = list(df.columns)
    align = align or ["---"] + ["---:"] * (len(cols) - 1)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(align) + "|"]
    for row in df.itertuples(index=False):
        out.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(out)


def _csv(name):
    p = FIG / name
    try:
        return pd.read_csv(p) if p.exists() else None
    except pd.errors.EmptyDataError:   # tabla sin filas (p. ej. en una prueba rápida)
        return None


def cargar() -> dict:
    R: dict = {}

    # ---------------- hardware ----------------
    hw = {}
    hp = OUT / "hardware.txt"
    if hp.exists():
        hw = {k.strip(): v.strip() for k, v in (l.split(":", 1) for l in hp.read_text().splitlines() if ":" in l)}
    cores = int(hw.get("CPU(s)", 0) or 0) // max(1, int(hw.get("Thread(s) per core", 1) or 1))
    R["nucleos"] = str(cores or "?")
    R["cpu"] = hw.get("Model name", "no registrado")
    R["equipo"] = (f"{R['cpu']}, {R['nucleos']} núcleos físicos, {hw.get('memoria', '?')} de RAM; {hw.get('so', '?')}; "
                   f"{hw.get('g++', 'g++ ?')}; {hw.get('mpi', 'MPI ?')}; {hw.get('python', 'Python ?')}; {hw.get('numpy', '')}")
    R["equipo_corto"] = f"{R['cpu']} ({R['nucleos']} núcleos, {hw.get('memoria', '?')} RAM)"
    R["fecha_medicion"] = hw.get("fecha", "?")[:10]
    R["gpu"] = hw.get("gpu", "no disponible")
    ne, nr = int(hw.get("nucleos_eficiencia", 0) or 0), int(hw.get("nucleos_rendimiento", cores) or cores)
    if ne > 0:
        R["equipo"] = R["equipo"].replace("núcleos físicos", f"núcleos físicos ({nr} de alto rendimiento y {ne} de eficiencia)")
        R["nota_nucleos"] = (
            f"El procesador es **heterogéneo**: {nr} núcleos de alto rendimiento y {ne} de eficiencia, sin hilos lógicos "
            f"adicionales. Las configuraciones con más de {nr} workers usan también los núcleos lentos. Con reparto "
            "estático (bloques iguales de filas en OpenMP, franjas iguales en MPI) el tiempo de cada sub-paso lo fija el "
            "núcleo más lento, de modo que la eficiencia cae por un motivo distinto a la fracción serial de Amdahl; por eso "
            f"el ajuste de Amdahl debe leerse sobre todo hasta p = {nr}, y el reparto dinámico cobra ventaja al usar todos "
            "los núcleos.")
    else:
        R["nota_nucleos"] = "Todos los núcleos del equipo son equivalentes (procesador homogéneo)."

    res, esc, val = _csv("tabla_resumen_tiempos.csv"), _csv("tabla_escalabilidad.csv"), _csv("tabla_validacion.csv")
    mej, pen, deb, var = (_csv("tabla_mejor_por_region.csv"), _csv("tabla_pendientes.csv"),
                          _csv("tabla_escalabilidad_debil.csv"), _csv("tabla_variabilidad.csv"))
    if res is None or esc is None or mej is None:
        return {**R, "_faltan_tablas": True}

    presentes = [p for p in NOMBRE if p in set(res.paradigm)]
    R["n_paradigmas"] = str(len(presentes))
    R["paradigmas"] = ", ".join(NOMBRE[p] for p in presentes)
    R["faltan"] = ", ".join(NOMBRE[p] for p in NOMBRE if p not in presentes) or "ninguno"

    # ---------------- región principal y p máximo válido ----------------
    regs = [r for r in ("pacifico", "andina", "caribe", "amazonia") if r in set(esc.region)]
    rp = regs[0]
    R["region_p"], R["region_p_nombre"] = rp, REGION[rp]
    ep = esc[(esc.region == rp) & (~esc.sobresuscrito)]
    R["L_p"] = str(int(res[res.region == rp].board_size.iloc[0]))
    pmax = int(ep.workers.max())
    R["pmax"] = str(pmax)

    def fila(par, w, reg=rp):
        q = esc[(esc.region == reg) & (esc.paradigma == par) & (esc.workers == w)]
        return q.iloc[0] if len(q) else None

    for par, k in (("openmp", "omp"), ("mpi", "mpi"), ("multiprocessing", "mp"), ("multiprocessing_franjas", "mpf")):
        q = ep[ep.paradigma == par]
        if q.empty:
            for s in ("S", "E", "f", "p", "kf"):
                R[f"{s}_{k}"] = "—"
            continue
        r = q.loc[q.workers.idxmax()]
        R[f"p_{k}"], R[f"S_{k}"], R[f"E_{k}"], R[f"f_{k}"] = str(int(r.workers)), n(r.speedup), n(r.eficiencia), n(r.amdahl_f, 3)
        kf = q[q.workers > 1].sort_values("workers")
        if len(kf) >= 2:
            a, b = kf.karp_flatt.iloc[0], kf.karp_flatt.iloc[-1]
            tend = "crece" if b > a + 0.01 else ("disminuye" if b < a - 0.01 else "se mantiene")
            R[f"kf_{k}"] = f"{tend} de {n(a, 3)} (p = {int(kf.workers.iloc[0])}) a {n(b, 3)} (p = {int(kf.workers.iloc[-1])})"
        else:
            R[f"kf_{k}"] = "—"

    fr = esc[(esc.paradigma == "multiprocessing_franjas") & (~esc.sobresuscrito)]
    if len(fr):
        reg_f = fr.region.iloc[0]
        q = fr[fr.region == reg_f]
        u = q.loc[q.workers.idxmax()]
        rep_ = esc[(esc.paradigma == "multiprocessing") & (esc.region == reg_f) & (esc.workers == u.workers)]
        R["franjas_resumen"] = (f"En {REGION[reg_f]}, con p = {int(u.workers)}, la estrategia de franjas alcanzó un speedup de "
                                f"{n(u.speedup)} (eficiencia {n(u.eficiencia)}, f de Amdahl {n(u.amdahl_f, 3)})"
                                + (f", frente a {n(rep_.speedup.iloc[0])} (eficiencia {n(rep_.eficiencia.iloc[0])}) del ensamble de réplicas"
                                   if len(rep_) else ""))
        R["franjas_S"], R["franjas_p"], R["franjas_E"] = n(u.speedup), str(int(u.workers)), n(u.eficiencia)
        R["franjas_region"] = REGION[reg_f]
    else:
        R["franjas_resumen"] = "[PENDIENTE: medir la estrategia de franjas con tools/benchmark_python_mp.sh]"
        R["franjas_S"] = R["franjas_p"] = R["franjas_E"] = R["franjas_region"] = "—"
    t4 = ep[ep.paradigma.isin(["openmp", "mpi", "multiprocessing", "multiprocessing_franjas"])].copy()
    R["tabla_escalabilidad"] = tabla_md(pd.DataFrame({
        "Paradigma": t4.paradigma.map(NOMBRE), "p": t4.workers.astype(int), "Tiempo (s)": t4["T (s)"].map(n),
        "Speedup": t4.speedup.map(n), "Eficiencia": t4.eficiencia.map(n),
        "Karp-Flatt e": t4.karp_flatt.map(lambda x: n(x, 3)), "f (Amdahl)": t4.amdahl_f.map(lambda x: n(x, 3))}))
    sob = esc[(esc.region == rp) & (esc.sobresuscrito)]
    R["sobresuscripcion"] = ("; ".join(f"{NOMBRE[r.paradigma]} con p = {int(r.workers)}: speedup {n(r.speedup)}" for r in sob.itertuples())
                             if len(sob) else "no se midieron configuraciones con más workers que núcleos")

    # ---------------- tiempos por región ----------------
    piv_t = mej.pivot(index="region", columns="paradigma", values="t")
    piv_w = mej.pivot(index="region", columns="paradigma", values="workers")
    orden = [r for r in REGION if r in piv_t.index]
    orden.sort(key=lambda r: mej[mej.region == r].L.iloc[0])
    cols = [p for p in NOMBRE if p in piv_t.columns]
    filas = []
    for r in orden:
        f = {"Región (L)": f"{REGION[r]} ({int(mej[mej.region == r].L.iloc[0])})"}
        for p in cols:
            t = piv_t.loc[r, p]
            w = piv_w.loc[r, p]
            f[NOMBRE[p]] = "—" if np.isnan(t) else (n(t) + (f" (p={int(w)})" if w > 1 else ""))
        filas.append(f)
    R["tabla_tiempos"] = tabla_md(pd.DataFrame(filas))

    def t(par, reg):
        try:
            return float(piv_t.loc[reg, par])
        except KeyError:
            return np.nan

    # speedup máximo respecto a C++ secuencial y respecto a NumPy
    best = (None, 0, None)
    for r in orden:
        for p in ("openmp", "mpi", "cuda", "cupy"):
            if p in cols and not np.isnan(t(p, r)) and not np.isnan(t("serial", r)):
                s = t("serial", r) / t(p, r)
                if s > best[1]:
                    best = (p, s, r)
    R["smax_serial"] = n(best[1]) if best[0] else "—"
    R["smax_serial_cfg"] = (f"{NOMBRE[best[0]]} con p = {int(piv_w.loc[best[2], best[0]])} en {REGION[best[2]]}" if best[0] else "—")
    bestn = (None, 0, None)
    for r in orden:
        for p in cols:
            if p not in ("numpy", "pure") and not np.isnan(t(p, r)) and not np.isnan(t("numpy", r)):
                s = t("numpy", r) / t(p, r)
                if s > bestn[1]:
                    bestn = (p, s, r)
    R["smax_numpy"] = n(bestn[1]) if bestn[0] else "—"
    R["smax_numpy_cfg"] = (f"{NOMBRE[bestn[0]]} con p = {int(piv_w.loc[bestn[2], bestn[0]])} en {REGION[bestn[2]]}" if bestn[0] else "—")
    R["t_serial_p"], R["t_numpy_p"] = n(t("serial", rp)), n(t("numpy", rp))
    ratio = t("serial", rp) / t("numpy", rp) if not np.isnan(t("numpy", rp)) else np.nan
    R["ratio_serial_numpy"] = n(ratio, 1)
    R["numpy_vs_serial"] = ("más rápido" if ratio > 1 else "más lento")
    R["numpy_vs_serial_inv"] = ("más lentas" if ratio > 1 else "más rápidas")
    rpure = [r for r in orden if "pure" in cols and not np.isnan(t("pure", r))]
    R["ratio_pure_serial"] = n(np.mean([t("pure", r) / t("serial", r) for r in rpure]), 1) if rpure else "—"
    R["ratio_pure_numpy"] = n(np.mean([t("pure", r) / t("numpy", r) for r in rpure]), 1) if rpure and "numpy" in cols else "—"
    R["regiones_pure"] = " y ".join(REGION[r] for r in rpure) or "—"
    # mejor paradigma de CPU compilado en la región más grande
    rg = orden[-1]
    R["region_grande"] = REGION[rg]
    R["ranking_grande"] = " < ".join(f"{NOMBRE[p]} ({n(v)} s)" for p, v in sorted(
        ((p, t(p, rg)) for p in cols if not np.isnan(t(p, rg))), key=lambda x: x[1]))
    R["ranking_p"] = " < ".join(f"{NOMBRE[p]} ({n(v)} s)" for p, v in sorted(
        ((p, t(p, rp)) for p in cols if not np.isnan(t(p, rp))), key=lambda x: x[1]))

    if pen is not None and len(pen):
        R["pendientes"] = "; ".join(f"{NOMBRE[r.paradigma]}: {n(r.pendiente)}" for r in pen.itertuples())
    else:
        R["pendientes"] = PEND

    ups = _csv("tabla_actualizaciones_por_segundo.csv")
    if ups is not None:
        g = ups.groupby("paradigma").actualizaciones_por_s.max()
        R["ups"] = "; ".join(f"{NOMBRE[p]}: {n(g[p] / 1e6, 2)} millones/s" for p in NOMBRE if p in g.index)

    # ---------------- GPU (medida en Google Colab) ----------------
    PG = "[PENDIENTE: ejecutar colab/GPU_en_Colab.ipynb y copiar outputs/gpu]"
    hgp = OUT / "gpu" / "hardware_gpu.txt"
    hg = {k.strip(): v.strip() for k, v in (l.split(":", 1) for l in hgp.read_text().splitlines() if ":" in l)} if hgp.exists() else {}
    R["gpu_nombre"] = hg.get("gpu", PG).split(",")[0]
    R["gpu_equipo"] = (f"{hg.get('gpu', '?')}; CuPy {hg.get('cupy', '?')}; CUDA runtime {hg.get('cuda_runtime', '?')}"
                       if hg else PG)
    gpus = [p for p in ("cupy", "cuda") if p in piv_t.columns]
    UNA_SIM = [p for p in ("serial", "openmp", "mpi", "numpy", "multiprocessing_franjas") if p in piv_t.columns]
    tg = _csv("tabla_gpu.csv")
    if gpus:
        filas_c, mejor_g = [], (None, None, np.inf)
        for r in orden:
            tcpu = min((t(p, r), p) for p in UNA_SIM if not np.isnan(t(p, r))) if UNA_SIM else (np.nan, None)
            for g in gpus:
                if not np.isnan(t(g, r)) and t(g, r) < mejor_g[2] and r == orden[-1]:
                    mejor_g = (g, r, t(g, r))
            tgpu = min((t(g, r), g) for g in gpus if not np.isnan(t(g, r))) if any(not np.isnan(t(g, r)) for g in gpus) else (np.nan, None)
            filas_c.append((r, tgpu, tcpu))
        gana = [r for r, tg_, tc in filas_c if tg_[0] < tc[0]]
        rg_ = orden[-1]
        g, _, tgv = mejor_g
        if g:
            best_cpu = min((t(p, rg_), p) for p in UNA_SIM if not np.isnan(t(p, rg_)))
            r_cpu = best_cpu[0] / tgv
            frase_cpu = (f"y {n(r_cpu)} veces más rápida que la mejor versión de CPU para una simulación ({NOMBRE[best_cpu[1]]})"
                         if r_cpu >= 1 else
                         f"pero {n(1 / r_cpu)} veces más lenta que la mejor versión de CPU para una simulación ({NOMBRE[best_cpu[1]]})")
            R["gpu_resumen"] = (f"En la GPU, la versión más rápida en {REGION[rg_]} (el tablero más grande medido) fue {NOMBRE[g]} "
                                f"({n(tgv)} s por simulación): {n(t('serial', rg_) / tgv)} veces más rápida que C++ secuencial, "
                                f"{frase_cpu}.")
        else:
            R["gpu_resumen"] = ""
        if gana:
            R["gpu_cruce"] = (f"La GPU supera a la mejor versión de CPU para una simulación a partir de "
                              f"{REGION[gana[0]]} (L = {int(mej[mej.region == gana[0]].L.iloc[0])})"
                              + (f"; en los tableros más pequeños ({', '.join(REGION[r] for r, *_ in filas_c if r not in gana)}) gana la CPU."
                                 if len(gana) < len(filas_c) else ", es decir, en todos los tamaños medidos."))
        else:
            R["gpu_cruce"] = (f"En los tamaños medidos (hasta L = {int(mej.L.max())}) la GPU no supera a la mejor versión de CPU para una simulación: "
                              "los tableros son demasiado pequeños para ocupar la GPU y domina el costo de lanzar kernels.")
        R["gpu_speedup_serial"] = n(t("serial", rg_) / tgv) if g else "—"
        R["gpu_cruce_corto"] = (f"La GPU gana a la CPU desde L = {int(mej[mej.region == gana[0]].L.iloc[0])} ({REGION[gana[0]]})"
                                if gana else f"Hasta L = {int(mej.L.max())}, la mejor CPU sigue ganando a la GPU")
        R["gpu_recomendacion"] = ("La GPU conviene a partir del tamaño de cruce indicado en la sección 7.3; por debajo, la CPU es más rápida y más simple."
                                  if gana else "Para estos tamaños, la GPU no compensa; se justificaría con tableros más finos (resolución municipal).")
        R["gpu_conclusion"] = R["gpu_cruce"]
    else:
        R["gpu_resumen"] = ""
        R["gpu_speedup_serial"] = "—"
        R["gpu_cruce_corto"] = "pendiente: ejecutar Colab"
        R["gpu_cruce"] = PG
        R["gpu_recomendacion"] = PG
        R["gpu_conclusion"] = PG
    if tg is not None and len(tg):
        R["tabla_gpu"] = tabla_md(pd.DataFrame({
            "Versión": tg.paradigm.map(NOMBRE), "L": tg.board_size.astype(int),
            "Copia a GPU (ms)": (tg.t_h2d * 1000).map(lambda x: n(x, 1)), "Cómputo (s)": tg.t_compute.map(n),
            "Copia de vuelta (ms)": (tg.t_d2h * 1000).map(lambda x: n(x, 1)),
            "Transferencias (% del total)": (100 * tg.fraccion_transferencia).map(lambda x: n(x, 2))}),
            align=["---", "---:", "---:", "---:", "---:", "---:"])
        lo, hi = 100 * tg.fraccion_transferencia.min(), 100 * tg.fraccion_transferencia.max()
        R["gpu_frac_max"] = n(hi, 2)
        R["gpu_transferencia"] = (f"Las transferencias entre CPU y GPU representan entre el {n(lo, 2)} % y el {n(hi, 2)} % del tiempo total, "
                                  "porque solo se copian el estado inicial y los resultados; el cruce lo determina el costo de lanzar "
                                  "kernels y el tamaño del tablero, no las copias.")
        R["gpu_texto"] = ("La Tabla 9b y la Figura 9b separan, para cada tamaño, la copia de los tableros a la GPU, el cómputo de los "
                          "204 meses y la copia de los resultados de vuelta (mediana de 5 repeticiones, medidas con eventos de CUDA). "
                          + R["gpu_transferencia"])
    else:
        R["tabla_gpu"] = ""
        R["gpu_frac_max"] = "—"
        R["gpu_transferencia"] = ""
        R["gpu_texto"] = PG

    # ---------------- escalabilidad débil ----------------
    if deb is not None and len(deb):
        R["tabla_debil"] = tabla_md(pd.DataFrame({
            "Procesos p": deb.workers.astype(int), "L": deb.board_size.astype(int),
            "Celdas por proceso": deb.celdas_por_proceso.round().astype(int), "Tiempo (s)": deb["median"].map(n),
            "Eficiencia débil": deb.eficiencia_debil.map(n), "Speedup escalado": deb.speedup_escalado.map(n)}),
            align=["---:"] * 6)
        u = deb.iloc[-1]
        R["debil_E"], R["debil_p"], R["debil_S"] = n(u.eficiencia_debil), str(int(u.workers)), n(u.speedup_escalado)
    else:
        R["tabla_debil"] = R["debil_E"] = R["debil_p"] = R["debil_S"] = PEND

    # ---------------- scheduling ----------------
    sp = OUT / "experiments" / "schedule" / "schedule.csv"
    if sp.exists():
        sc = pd.read_csv(sp)
        g = sc.groupby(["region", "schedule"]).elapsed_time.median().unstack()
        g = g.reindex([r for r in REGION if r in g.index])
        R["sched_hilos"] = str(int(sc.threads.iloc[0]))
        R["tabla_schedule"] = tabla_md(pd.DataFrame({
            "Región": [REGION[r] for r in g.index], **{c: g[c].map(n).values for c in ("static", "dynamic", "guided") if c in g}}))
        gp = g.loc[rp] if rp in g.index else g.iloc[-1]
        mejor_s = gp.idxmin()
        dif = 100 * (gp["static"] - gp[mejor_s]) / gp["static"]
        R["sched_mejor"], R["sched_dif"] = mejor_s, n(dif, 0)
        if mejor_s == "static" or dif < 3:
            R["sched_frase"] = ("la medición confirmó la elección: `static` fue el más rápido o quedó a menos de un 3 % del mejor, "
                                "como corresponde a filas de costo uniforme")
        else:
            R["sched_frase"] = (f"`{mejor_s}` fue un {n(dif, 0)} % más rápido que `static` en {REGION[rp]}, lo que indica que la carga "
                                "por fila **no** es uniforme: el costo del muestreo binomial depende del estado epidémico de cada "
                                "celda y, en un equipo compartido, de la disponibilidad de cada núcleo")
    else:
        R["tabla_schedule"] = R["sched_frase"] = R["sched_mejor"] = R["sched_dif"] = R["sched_hilos"] = PEND

    # ---------------- lgamma ----------------
    lp, tp = OUT / "experiments" / "times_cpp_glibc_lgamma.csv", OUT / "times.csv"
    if lp.exists() and tp.exists():
        a, b = pd.read_csv(lp), pd.read_csv(tp)
        filas = []
        for r in [x for x in REGION if x in set(a[a.paradigm == "openmp"].region) and x in set(b[b.paradigm == "openmp"].region)]:
            s1 = b[(b.paradigm == "serial") & (b.region == r)].elapsed_time.median()
            s1a = a[(a.paradigm == "serial") & (a.region == r)].elapsed_time.median()
            s1a = s1 if np.isnan(s1a) else s1a
            qa = a[(a.paradigm == "openmp") & (a.region == r) & (a.workers <= cores if cores else True)].groupby("workers").elapsed_time.median()
            qb = b[(b.paradigm == "openmp") & (b.region == r) & (b.workers <= cores if cores else True)].groupby("workers").elapsed_time.median()
            w = min(qa.index.max(), qb.index.max())
            filas.append({"Región": REGION[r], "Hilos": int(w), "Speedup con `lgamma` (estado global)": n(s1a / qa[w]),
                          "Speedup con `lgamma_r` (reentrante)": n(s1 / qb[w]), "_a": s1a / qa[w], "_b": s1 / qb[w], "_r": r})
        df = pd.DataFrame(filas)
        R["tabla_lgamma"] = tabla_md(df[[c for c in df.columns if not c.startswith("_")]])
        f0 = df[df._r == rp].iloc[0] if rp in set(df._r) else df.iloc[-1]
        R["lgamma_antes"], R["lgamma_despues"], R["lgamma_hilos"] = n(f0._a), n(f0._b), str(f0.Hilos)
    else:
        R["tabla_lgamma"] = R["lgamma_antes"] = R["lgamma_despues"] = R["lgamma_hilos"] = PEND

    # ---------------- validación ----------------
    try:
        import sys
        sys.path.insert(0, str(ROOT / "tools"))
        from validate_metrics import check
        files = sorted((OUT / "metrics").glob("*.csv"))
        R["n_archivos"], R["n_validos"] = str(len(files)), str(sum(1 for f in files if not check(f)))
    except Exception:
        R["n_archivos"] = R["n_validos"] = "—"
    if val is not None and len(val):
        v = val.copy()
        R["tabla_validacion"] = tabla_md(pd.DataFrame({
            "Región": v.region.map(REGION), "Especie": "*P. " + v.cepa + "*",
            "Implementación": v.paradigma.map(lambda p: NOMBRE[p] + (" (núcleo NumPy)" if p == "multiprocessing" else "")),
            "Réplicas (impl. / ref.)": v.n.astype(str) + " / " + v.n_ref.astype(str),
            "Diferencia del tamaño final (%)": v["dif_relativa_%"].map(lambda x: ("+" if x > 0 else "") + n(x, 2 if abs(x) < 10 else 0)),
            "¿Dentro de 2 SE?": v.cumple_2SE.map({True: "sí", False: "no"}),
            "Meses dentro de la banda (%)": v["%_meses_en_banda"].map(lambda x: n(x, 0))}),
            align=["---", "---", "---", "---:", "---:", ":---:", "---:"])
        q = v[v.paradigma.isin(["multiprocessing", "numpy"])]
        peor = q.loc[q["dif_relativa_%"].abs().idxmax()] if len(q) else None
        if peor is not None:
            R["numpy_peor"] = (f"*P. {peor.cepa}* en {REGION[peor.region]}: {('+' if peor['dif_relativa_%'] > 0 else '')}"
                               f"{n(peor['dif_relativa_%'], 0)} % en el tamaño final y {n(peor['%_meses_en_banda'], 0)} % de los meses dentro de la banda")
            resto = q.drop(peor.name)["dif_relativa_%"].abs()
            R["numpy_resto"] = f"entre {n(resto.min(), 2)} % y {n(resto.max(), 2)} %" if len(resto) else "—"
        c = v[v.paradigma == "serial"]
        R["cpp_cumple"] = f"{int(c.cumple_2SE.sum())} de {len(c)}"
        R["cpp_dif_max"] = n(c["dif_relativa_%"].abs().max(), 2)
        R["cpp_banda_min"] = n(c["%_meses_en_banda"].min(), 0)
        cv_ = c[(c.cepa == "vivax")]
        R["cpp_vivax_dif"] = "; ".join(f"{REGION[r.region]}: {('+' if r._4 > 0 else '')}{n(r._4, 3)} %" for r in
                                       cv_[["region", "cepa", "paradigma", "n", "dif_relativa_%"]].itertuples(index=False)) if len(cv_) else "—"
    else:
        for k in ("tabla_validacion", "numpy_peor", "numpy_resto", "cpp_cumple", "cpp_dif_max", "cpp_banda_min", "cpp_vivax_dif"):
            R[k] = PEND
    if var is not None and len(var):
        ref = var[var.paradigma.isin(["pure", "serial"])].sort_values("cv_%", ascending=False).iloc[0]
        R["var_caso"] = f"*P. {ref.cepa}* en {REGION[ref.region]}"
        R["var_cv"] = n(ref["cv_%"], 0)
        est = var[(var.paradigma == ref.paradigma) & (var.cepa == "vivax") & (var.region == ref.region)]
        R["var_cv_estable"] = n(est["cv_%"].iloc[0], 2) if len(est) else "—"
        ex = var[(var.p_extincion > 0) & var.paradigma.isin(["pure", "serial"])]
        R["extinciones"] = ("; ".join(f"*P. {r.cepa}* en {REGION[r.region]} (probabilidad {n(r.p_extincion, 2)})" for r in ex.itertuples())
                            or "en ninguna región evaluada")
    else:
        R["var_caso"] = R["var_cv"] = R["var_cv_estable"] = R["extinciones"] = PEND
    return R


if __name__ == "__main__":
    for k, v in cargar().items():
        print(f"--- {k}\n{v}")
