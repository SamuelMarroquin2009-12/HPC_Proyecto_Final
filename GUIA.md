# Guía del proyecto (en lenguaje sencillo)

Esta guía explica, sin tecnicismos innecesarios, qué hay en el proyecto, cómo ejecutarlo y
qué hay que saber para la sustentación.

---

## 1. ¿De qué trata el proyecto?

Tenemos un **simulador de malaria** para las cinco regiones de Colombia, mes a mes, de 2007 a
2023. Cada región es una cuadrícula: cada cuadrito representa 1 000 personas y hasta 2 000
mosquitos, y "habla" con sus 8 vecinos. El clima de cada mes decide cuántos mosquitos nacen y
cuánto contagian. Como hay azar, cada ejecución da un resultado un poco distinto.

El curso no evalúa el modelo: evalúa **cómo lo hacemos correr más rápido** y si sabemos
explicar por qué. Por eso el mismo simulador está escrito de **8 formas**:

| Versión | En palabras sencillas |
|---|---|
| Python puro | Recorre las celdas una por una. Es lenta, pero es la referencia de "lo correcto". |
| NumPy | Calcula el tablero completo de una vez, como una hoja de cálculo. |
| multiprocessing | Varios procesos de Python a la vez. Dos formas: cada proceso corre simulaciones enteras (**réplicas**), o todos se reparten un mismo tablero por filas (**franjas**). |
| CuPy | La versión NumPy ejecutada en la tarjeta gráfica (GPU). |
| C++ secuencial | El mismo algoritmo, compilado. Es la base justa para comparar. |
| OpenMP | C++ con varios hilos que comparten la memoria; cada hilo hace unas filas del tablero. |
| MPI | C++ con varios procesos que no comparten memoria: cada uno tiene un pedazo del tablero y se pasan solo los bordes. |
| Kernel CUDA | Programa propio para la GPU: un hilo por cada celda. |

**Crédito:** el modelo base lo hizo Daniel Solarte (antiguo integrante) en otro curso; se usa
con licencia MIT y su nombre se mantiene en `LICENSE` y en el README, como exige esa licencia.

---

## 2. Preparación (una sola vez)

Todo se ejecuta en **WSL (Ubuntu)** dentro de Windows. Abra la terminal de Ubuntu (o en VS Code:
Terminal → nueva terminal → Ubuntu (WSL)) y ejecute, uno por uno:

```bash
sudo apt update && sudo apt install -y build-essential openmpi-bin libopenmpi-dev pandoc python3-venv python3-pip zip
```

```bash
cd /mnt/c/Users/SAMUEL/HPC_Proyecto_Final
```

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

Cada vez que abra una terminal nueva, active el entorno antes de trabajar:

```bash
cd /mnt/c/Users/SAMUEL/HPC_Proyecto_Final && source .venv/bin/activate
```

---

## 3. Ejecutar todo (en el portátil)

**Primero una prueba rápida** (unos 5 minutos) para comprobar que todo está bien instalado:

```bash
RAPIDO=1 bash tools/run_all.sh
```

Si termina con `LISTO`, haga la **medición completa** (cerca de 1,5 horas; deje el portátil
conectado a la corriente y sin usar otros programas pesados, porque eso altera los tiempos):

```bash
bash tools/run_all.sh
```

El script hace todo solo: compila, mide cada versión 5 veces, valida que todas den el mismo
resultado, ejecuta el notebook (todas las gráficas), genera `informe.pdf` y
`docs/presentacion.pptx`.

> **Sobre los núcleos:** el i5-1135G7 tiene **4 núcleos físicos** y 8 hilos (Windows muestra 8).
> La guía del curso dice que no se usen más procesos que núcleos físicos, así que las mediciones
> válidas son con 1, 2 y 4. El script también mide 8 hilos en OpenMP y multiprocessing, pero
> solo como punto de comparación "sobresuscrito" (en las gráficas aparece hueco y no entra en
> los cálculos). Sirve para mostrar que usar más hilos que núcleos no acelera.

---

## 4. Ejecutar la parte de GPU (Google Colab)

El portátil tiene gráfica Intel, y CuPy y CUDA necesitan una NVIDIA. La guía permite usar
Google Colab, que presta una gratis:

1. Entre a <https://colab.research.google.com> → **Archivo → Abrir cuaderno → GitHub**, escriba
   `SamuelMarroquin2009-12/HPC_Proyecto_Final` y abra `colab/GPU_en_Colab.ipynb`.
2. **Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU → Guardar.**
3. **Entorno de ejecución → Ejecutar todas.** Tarda unos minutos. Al final se descarga
   `resultados_gpu.zip`.
4. Copie el zip a la carpeta `outputs` del proyecto y descomprímalo ahí, de modo que quede
   `outputs/gpu/times.csv`. Desde la terminal:

```bash
cd /mnt/c/Users/SAMUEL/HPC_Proyecto_Final/outputs && unzip -o /mnt/c/Users/SAMUEL/Downloads/resultados_gpu.zip && cd ..
```

5. Regenere el notebook, el informe y la presentación con los datos de GPU (no vuelve a medir la CPU):

```bash
bash tools/run_all.sh --solo-informe
```

> El repositorio debe estar **público** para que Colab lo pueda descargar (o suba el cuaderno a
> mano con Archivo → Subir cuaderno y cambie la línea `git clone` por la de su repo).

---

## 5. Subir los resultados a GitHub

Cuando ya tenga las mediciones (CPU y GPU), súbalas:

```bash
git add -A && git commit -m "Resultados medidos en el portátil (i5-1135G7) y en Google Colab (GPU)" && git push
```

---

## 6. Qué revisar antes de entregar

1. `informe.pdf`: busque las marcas **amarillas** (lo que falta). Normalmente quedan:
   - las **horas de trabajo** de la Tabla 10 (pónganlas ustedes);
   - las **referencias por verificar** (fuente oficial del dataset, guía del curso).
   Esas se editan en `docs/informe_plantilla.md` (es texto normal) y luego:
   `bash tools/run_all.sh --solo-informe`.
2. `docs/presentacion.pptx`: ábrala y repásela; las notas del orador de cada diapositiva dicen
   qué explicar.
3. `docs/presentacion.md`: tiene el guion y **15 preguntas probables del profesor con sus
   respuestas**. Estúdienlas los dos.

---

## 7. Lo que hay que saber explicar (resumen para la sustentación)

- **Speedup** = tiempo de referencia ÷ tiempo paralelo. **Eficiencia** = speedup ÷ número de
  núcleos (1 = perfecto).
- **Referencias:** el speedup se mide contra C++ secuencial (comparación justa entre técnicas)
  y contra NumPy (lo que gana alguien que usa Python).
- **Amdahl:** lo que no se puede paralelizar limita el speedup. **Gustafson:** si el problema
  crece con los núcleos, el speedup escala mejor.
- **Cómo sabemos que todo está bien:** (1) en cada mes la población se conserva; (2) C++
  secuencial, OpenMP y MPI dan **exactamente** el mismo resultado; (3) entre lenguajes se
  comparan promedios de varias ejecuciones (deben coincidir dentro del error estadístico).
- **Hallazgos propios (los más valiosos):**
  1. OpenMP iba más lento de lo esperado porque una función matemática de la biblioteca
     (`lgamma`) usa una variable global que todos los hilos tocan; se cambió por `lgamma_r`.
  2. La versión NumPy original muestreaba mal cuando había pocos casos y exageraba las epidemias
     cerca del umbral (hasta 7 veces más casos en Pacífico); la validación lo detectó y se
     corrigió usando la binomial exacta.
  3. La librería estándar de C++ tiene un sesgo minúsculo (0,04 %) en su binomial; la
     validación también lo detectó.
- **GPU:** en tableros pequeños la GPU no gana, porque lanzar cada operación cuesta más que
  calcularla; gana a partir de cierto tamaño (lo dice el informe con los datos). Las copias
  entre CPU y GPU pesan poco, porque solo se hacen al inicio y al final.

---

## 8. Demostración en vivo (guía §6.8)

La guía pide mostrar en vivo al menos la versión CuPy y una versión compilada:

```bash
cd cpp && make && mpirun -np 4 ./mpi --region amazonia --warmup 0 && cd ..
```

En Colab, con el cuaderno abierto, ejecute una celda con:
`!python main.py cupy --region amazonia --runs 1 --disable-animation`

---

## 9. ¿Dónde está cada cosa? (correspondencia con la guía)

Ver [docs/CORRESPONDENCIA_CON_LA_GUIA.md](./docs/CORRESPONDENCIA_CON_LA_GUIA.md). En resumen:
`01_secuencial.py` → `python main.py pure`, `02_numpy.py` → `python main.py numpy`,
`03_paralelo_cpu.py` → `python main.py multiprocessing`, `04_cupy.py` → `python main.py cupy`,
`05_cupy_rawkernel.py` → `python main.py cuda`, `resultados/` → `outputs/`.
