// =============================================================================
// kernel_automata.cu — Kernel CUDA del autómata celular (un hilo por celda)
//
// Avanza UN sub-paso de UNA cepa: lee el estado actual (5 tableros) y escribe el
// siguiente (doble buffer), igual que ac_step_strain_day de utils_pure.py.
// Se compila en tiempo de ejecución con cupy.RawKernel (ver utils_cuda.py).
//
// Números aleatorios: generador sin estado por hilo. Cada hilo deriva su semilla de
// (flujo del sub-paso, índice de celda) con SplitMix64 y genera su secuencia; el
// "flujo" lo calcula el host a partir de (semilla, réplica, mes, sub-paso, cepa).
// Muestreo Binomial EXACTO: inversión si n·q < 10 y BTRS (Hörmann, 1993) si no,
// con q = min(p, 1 - p) y la simetría Bin(n, p) = n - Bin(n, 1 - p).
//
// El mismo archivo compila también en CPU (g++ -DEMULAR_CPU) para poder probar la
// lógica sin GPU: tools/test_kernel_cpu.py.
// =============================================================================
#ifdef EMULAR_CPU
#include <math.h>
static int g_tid;                       // índice del "hilo" en la emulación
#define HILO_ID g_tid
#define DISPOSITIVO static inline
#define KERNEL extern "C"
#else
#define HILO_ID (blockIdx.x * blockDim.x + threadIdx.x)
#define DISPOSITIVO __device__ __forceinline__
#define KERNEL extern "C" __global__
#endif

typedef unsigned long long u64;

DISPOSITIVO u64 splitmix64(u64* s) {
    u64 z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Uniforme en (0, 1)
DISPOSITIVO double uniforme(u64* s) {
    return ((double)(splitmix64(s) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

// Binomial por inversión (n·p pequeño, p <= 0,5)
DISPOSITIVO int binomial_inversion(int n, double p, u64* s) {
    const double q = 1.0 - p;
    const double sp = p / q;
    const double a = (n + 1) * sp;
    double r = pow(q, (double)n);
    double u = uniforme(s);
    int x = 0;
    while (u > r && x < n) {
        u -= r;
        x++;
        r *= (a / x - sp);
        if (r <= 0.0) break;
    }
    return x;
}

// Binomial por BTRS (transformed rejection with squeeze), n·p >= 10, p <= 0,5
DISPOSITIVO int binomial_btrs(int n, double p, u64* s) {
    const double q = 1.0 - p;
    const double spq = sqrt(n * p * q);
    const double b = 1.15 + 2.53 * spq;
    const double a = -0.0873 + 0.0248 * b + 0.01 * p;
    const double c = n * p + 0.5;
    const double v_r = 0.92 - 4.2 / b;
    const double alpha = (2.83 + 5.1 / b) * spq;
    const double lpq = log(p / q);
    const double m = floor((n + 1) * p);
    const double h = lgamma(m + 1.0) + lgamma(n - m + 1.0);
    for (int intento = 0; intento < 1000; ++intento) {
        const double u = uniforme(s) - 0.5;
        double v = uniforme(s);
        const double us = 0.5 - fabs(u);
        const double k = floor((2.0 * a / us + b) * u + c);
        if (k < 0.0 || k > n) continue;
        if (us >= 0.07 && v <= v_r) return (int)k;
        v = log(v * alpha / (a / (us * us) + b));
        const double cota = h - lgamma(k + 1.0) - lgamma(n - k + 1.0) + (k - m) * lpq;
        if (v <= cota) return (int)k;
    }
    return (int)floor(n * p + 0.5);  // salvaguarda (prácticamente inalcanzable)
}

DISPOSITIVO int binomial(int n, double p, u64* s) {
    if (n <= 0 || p <= 0.0) return 0;
    if (p >= 1.0) return n;
    const int simetrico = p > 0.5;
    const double q = simetrico ? 1.0 - p : p;
    const int x = (n * q < 10.0) ? binomial_inversion(n, q, s) : binomial_btrs(n, q, s);
    return simetrico ? n - x : x;
}

DISPOSITIVO double prob_infeccion(double tasa, double exposicion, double dt) {
    const double e = exposicion * dt;
    if (e <= 0.0 || tasa <= 0.0) return 0.0;
    if (tasa >= 1.0) return 1.0;
    double r = 1.0 - pow(1.0 - tasa, e);
    return r < 0.0 ? 0.0 : (r > 1.0 ? 1.0 : r);
}

KERNEL void paso_cepa(
    const int* Sh, const int* Ih, const int* Rh, const int* Sm, const int* Im,
    int* nSh, int* nIh, int* nRh, int* nSm, int* nIm, int* casos,
    const int L,
    const double tasa_m_h, const double tasa_h_m,
    const double p_rec, const double p_perdida, const double p_recaida, const int hay_recaida,
    const double p_muerte, const double idoneidad, const double dt,
    const u64 flujo)
{
    const int i = HILO_ID;
    if (i >= L * L) return;
    const int r = i / L, c = i % L;
    const int ru = (r - 1 + L) % L, rd = (r + 1) % L;
    const int cl = (c - 1 + L) % L, cr = (c + 1) % L;
    const int vec[8] = {ru * L + cl, ru * L + c, ru * L + cr, r * L + cl, r * L + cr, rd * L + cl, rd * L + c, rd * L + cr};

    int ih_v = 0, im_v = 0, vivos_v = 0;
    for (int k = 0; k < 8; ++k) {
        ih_v += Ih[vec[k]];
        im_v += Im[vec[k]];
        vivos_v += Sm[vec[k]] + Im[vec[k]];
    }

    u64 s = flujo ^ ((u64)i * 0xD1B54A32D192ED03ULL);
    splitmix64(&s);

    const int s_h = Sh[i], i_h = Ih[i], r_h = Rh[i], s_m = Sm[i], i_m = Im[i];
    const double alfa = 0.25, N_H = 1000.0, K_M = 2000.0;
    const double im_ef = ((1.0 - alfa) * i_m + alfa * (im_v / 8.0)) / N_H;
    const double ih_ef = ((1.0 - alfa) * i_h + alfa * (ih_v / 8.0)) / N_H;

    // Humanos: S -> I -> R -> S (+ recaídas en P. vivax)
    const int nuevos_i = binomial(s_h, prob_infeccion(tasa_m_h, im_ef, dt), &s);
    const int nuevos_r = binomial(i_h, p_rec, &s);
    const int recaidas = hay_recaida ? binomial(r_h, p_recaida, &s) : 0;
    const int r_rest = r_h - recaidas > 0 ? r_h - recaidas : 0;
    const int pierden = binomial(r_rest, p_perdida, &s);
    nSh[i] = s_h - nuevos_i + pierden;
    nIh[i] = i_h + nuevos_i - nuevos_r + recaidas;
    nRh[i] = r_rest + nuevos_r - pierden;
    casos[i] = nuevos_i + recaidas;

    // Mosquitos: nacimiento -> S -> I -> muerte
    const int muertos_s = binomial(s_m, p_muerte, &s);
    const int muertos_i = binomial(i_m, p_muerte, &s);
    const int s_m_v = s_m - muertos_s, i_m_v = i_m - muertos_i;
    int vacios = (int)K_M - (s_m_v + i_m_v);
    if (vacios < 0) vacios = 0;
    double p_nac = (0.12 * idoneidad + 0.04 * ((vivos_v / 8.0) / K_M)) * dt;
    p_nac = p_nac < 0.0 ? 0.0 : (p_nac > 1.0 ? 1.0 : p_nac);
    const int nacen = binomial(vacios, p_nac, &s);
    const int infectados_m = binomial(s_m_v, prob_infeccion(tasa_h_m, ih_ef, dt), &s);
    nSm[i] = s_m_v - infectados_m + nacen;
    nIm[i] = i_m_v + infectados_m;
}

#ifdef EMULAR_CPU
// Emulación en CPU: recorre todas las celdas como si fueran hilos.
extern "C" void lanzar_cpu(
    int hilos,
    const int* Sh, const int* Ih, const int* Rh, const int* Sm, const int* Im,
    int* nSh, int* nIh, int* nRh, int* nSm, int* nIm, int* casos,
    int L, double tasa_m_h, double tasa_h_m, double p_rec, double p_perdida, double p_recaida,
    int hay_recaida, double p_muerte, double idoneidad, double dt, u64 flujo)
{
    for (g_tid = 0; g_tid < hilos; ++g_tid)
        paso_cepa(Sh, Ih, Rh, Sm, Im, nSh, nIh, nRh, nSm, nIm, casos, L, tasa_m_h, tasa_h_m,
                  p_rec, p_perdida, p_recaida, hay_recaida, p_muerte, idoneidad, dt, flujo);
}
#endif
