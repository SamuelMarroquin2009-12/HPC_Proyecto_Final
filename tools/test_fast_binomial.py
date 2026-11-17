"""
Prueba del muestreador vectorizado de NumPy/CuPy (python/adapters/utils_npcp.ac_fast_binomial).
Casos donde la aproximación inicial fallaba (media baja junto a celdas de media alta, y p
grande con n pequeño). También prueba el respaldo aproximado corregido.

Caso: un tablero donde casi todas las celdas tienen media n*p = 0,5 y UNA celda tiene media
alta. La versión inicial elegía la aproximación con el máximo del tablero y daba
P(x > 0) ≈ 0,50 en esas celdas; la exacta es 1 - (1 - p)^n ≈ 0,39.

Uso (desde la raíz del repo):  python tools/test_fast_binomial.py
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from adapters.utils_npcp import ac_fast_binomial  # noqa: E402

from adapters.utils_npcp import _binomial_aproximada  # noqa: E402

rng = np.random.default_rng(1)
ok = True
for nombre, f in (("exacta", ac_fast_binomial), ("respaldo aproximado", _binomial_aproximada)):
    n = np.full((300, 300), 1000, np.int32)
    p = np.full((300, 300), 0.0005)
    p[0, 0] = 0.5
    x = f(np, rng, n, p)[1:, :]
    exacta = 1 - (1 - 0.0005) ** 1000
    print(f"[{nombre}] media baja: media {x.mean():.3f} (0,500), P(x>0) {(x > 0).mean():.3f} ({exacta:.3f})")
    if nombre == "exacta":
        ok &= abs(x.mean() - 0.5) < 0.01 and abs((x > 0).mean() - exacta) < 0.02
    for n_, p_ in ((5, 0.53), (8, 0.53)):
        y = f(np, rng, np.full((400, 400), n_, np.int32), np.full((400, 400), p_))
        print(f"[{nombre}] n={n_}, p={p_}: media {y.mean():.3f} ({n_ * p_:.3f})")
        if nombre == "exacta":
            ok &= abs(y.mean() - n_ * p_) < 0.02
print("OK (la exacta coincide; el respaldo aproximado solo se informa)" if ok else "FALLA")
sys.exit(0 if ok else 1)
