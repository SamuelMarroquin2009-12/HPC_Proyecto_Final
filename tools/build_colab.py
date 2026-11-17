"""
Genera colab/GPU_en_Colab.ipynb: ejecuta las versiones de GPU (CuPy y kernel CUDA) en
Google Colab y descarga los resultados en un zip para copiarlos a outputs/gpu/ del repo.

Uso (desde la raíz del repo):  python tools/build_colab.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
REPO = "https://github.com/SamuelMarroquin2009-12/HPC_Proyecto_Final.git"
c = []
md = lambda t: c.append(nbf.v4.new_markdown_cell(t.strip()))  # noqa: E731
code = lambda t: c.append(nbf.v4.new_code_cell(t.strip()))  # noqa: E731

md("""
# Versiones de GPU en Google Colab (CuPy y kernel CUDA)

El computador del equipo no tiene GPU NVIDIA, así que estas dos versiones se miden aquí
(la guía lo permite, §1.5). Pasos:

1. **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU (T4)** y Guardar.
2. **Entorno de ejecución → Ejecutar todas.** Tarda unos minutos.
3. Al final se descarga `resultados_gpu.zip`. Descomprímalo dentro de la carpeta `outputs/`
   del repositorio (debe quedar `outputs/gpu/times.csv`, `outputs/gpu/times_gpu.csv`, ...).
4. En el computador, regenere el análisis: `bash tools/run_all.sh --solo-informe`.
""")
md("## 1. Comprobar la GPU")
code("!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv")
md("## 2. Descargar el proyecto")
code(f"""
import os
if not os.path.exists("HPC_Proyecto_Final"):
    !git clone -q {REPO}
%cd HPC_Proyecto_Final/python
!pip -q install pyarrow
import cupy as cp
print("CuPy", cp.__version__, "| GPU:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
""")
md("""
## 3. Medir CuPy y el kernel CUDA en las 5 regiones

Cada comando hace 1 ejecución de calentamiento (compila los kernels) y 5 repeticiones.
Por cada repetición se mide aparte la copia a la GPU, el cómputo y la copia de vuelta,
sincronizando con eventos de CUDA (guía §6.4).
""")
code("""
import os
os.environ["HPC_OUTPUTS"] = "../outputs/gpu"
for paradigma in ("cupy", "cuda"):
    for region in ("amazonia", "orinoquia", "pacifico", "caribe", "andina"):
        print(f"=== {paradigma} · {region} ===", flush=True)
        !python main.py {paradigma} --region {region} --runs 5 --warmup 1 --disable-animation 2>&1 | grep -E "TIEMPO|ERROR|Error|Traceback"
""")
md("""
## 4. Réplicas para la validación estadística

8 réplicas de Amazonía y Pacífico (las mismas regiones que la referencia de Python puro).
""")
code("""
for paradigma in ("cupy", "cuda"):
    for region in ("amazonia", "pacifico"):
        !python main.py {paradigma} --region {region} --runs 8 --warmup 0 --seed 5000 --disable-animation 2>&1 | grep -cE "TIEMPO"
""")
md("## 5. Registrar el equipo y descargar los resultados")
code("""
import subprocess, platform
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
with open("../outputs/gpu/hardware_gpu.txt", "w") as f:
    f.write(f"gpu: {gpu}\\n")
    f.write(f"cupy: {cp.__version__}\\n")
    f.write(f"cuda_runtime: {cp.cuda.runtime.runtimeGetVersion()}\\n")
    f.write(f"cpu_host: {platform.processor() or platform.machine()}\\n")
    f.write("equipo: Google Colab\\n")
print(open("../outputs/gpu/hardware_gpu.txt").read())
!cd ../outputs && rm -f ../resultados_gpu.zip && zip -qr ../resultados_gpu.zip gpu
from google.colab import files
files.download("../resultados_gpu.zip")
""")
nb = nbf.v4.new_notebook(cells=c, metadata={"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                                            "kernelspec": {"display_name": "Python 3", "name": "python3"}})
(ROOT / "colab").mkdir(exist_ok=True)
nbf.write(nb, ROOT / "colab" / "GPU_en_Colab.ipynb")
print("escrito colab/GPU_en_Colab.ipynb")
