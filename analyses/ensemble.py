"""
Roda o ensemble do modelo de vacinação e salva um arquivo .npz por pedaço.

Um "pedaço" é um intervalo de seeds de um cenário. Cada pedaço vira um
arquivo ``<saida>/<código>_seeds<ini>-<fim>.npz``; pedaços que já existem são
pulados, então um job interrompido pode ser simplesmente relançado. Depois de
rodar tudo, ``analyses/juntar_pedacos.py`` reúne os pedaços em
``results/<código>.npz``.

Exemplos, a partir da raiz do repositório::

    # tudo, em 8 processos, numa máquina só
    python analyses/ensemble.py --processos 8

    # quantas tarefas (cenário, pedaço) existem com este tamanho de pedaço
    python analyses/ensemble.py --tamanho-pedaco 50 --listar

    # uma única tarefa, como num job array do SLURM
    python analyses/ensemble.py --tamanho-pedaco 50 --tarefa 17

    # teste rápido: 3 seeds, 400 passos
    python analyses/ensemble.py --seed-fim 3 --horizonte 400 --saida /tmp/teste/pedacos
"""
import os

# Uma thread de BLAS/OpenMP por processo: o paralelismo é por seed, não por
# thread. Precisa vir antes de importar o NumPy.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model_vacinacao import CENARIOS, parametros_cenario, simular

# Chaves com uma entrada por seed (primeiro eixo = seeds do pedaço).
CHAVES_POR_SEED = ("n_inf_faixa", "n_naoinf_faixa", "novas_inf_faixa", "trans_faixa",
                   "doses1", "reforcos", "imunidade_media", "hist_infeccoes",
                   "t_extincao", "t_parada", "tempo_s")
# Chaves das grades, guardadas só para as seeds de --seeds-grades.
CHAVES_GRADES = ("grades_m", "grades_infectado")


def tempos_grades(horizonte):
    """Passos em que as grades são guardadas: 0, 60, 300 e o horizonte."""
    return sorted({t for t in (0, 60, 300, horizonte) if t <= horizonte})


def rodar_seed(args):
    """Roda uma execução. O resultado depende só de (cenário, seed, horizonte)."""
    codigo, seed, horizonte, guarda_grades = args
    p = parametros_cenario(codigo, horizonte=horizonte)
    t0 = time.perf_counter()
    res = simular(p, seed, snapshots=tempos_grades(horizonte) if guarda_grades else ())
    res["tempo_s"] = np.float32(time.perf_counter() - t0)
    return res


def rodar_pedaco(codigo, seeds, horizonte, seeds_grades=(0,), executor=None):
    """Roda as ``seeds`` de um cenário e empilha os resultados em arrays.

    Com ``executor`` (um ProcessPoolExecutor), as seeds rodam em paralelo; o
    resultado é idêntico ao da execução em série.
    """
    seeds = [int(s) for s in seeds]
    tarefas = [(codigo, s, horizonte, s in seeds_grades) for s in seeds]
    resultados = list(executor.map(rodar_seed, tarefas)) if executor else [rodar_seed(a) for a in tarefas]

    dados = {k: np.stack([r[k] for r in resultados]) for k in CHAVES_POR_SEED}
    com_grades = [i for i, s in enumerate(seeds) if s in seeds_grades]
    N = parametros_cenario(codigo).N
    n_t = len(tempos_grades(horizonte))
    dados["grades_seeds"] = np.array([seeds[i] for i in com_grades], dtype=np.int32)
    dados["grades_tempos"] = np.array(tempos_grades(horizonte), dtype=np.int32)
    dados["grades_m"] = (np.stack([resultados[i]["grades_m"] for i in com_grades]) if com_grades
                         else np.zeros((0, n_t, N, N), dtype=np.float32))
    dados["grades_infectado"] = (np.stack([resultados[i]["grades_infectado"] for i in com_grades])
                                 if com_grades else np.zeros((0, n_t, N, N), dtype=bool))
    dados["seeds"] = np.array(seeds, dtype=np.int32)
    dados["codigo"] = np.array(codigo)
    dados["horizonte"] = np.int32(horizonte)
    dados["parametros_json"] = np.array(parametros_cenario(codigo, horizonte=horizonte).para_json())
    return dados


def nome_pedaco(codigo, ini, fim):
    """Nome do arquivo do pedaço com as seeds ``ini`` a ``fim - 1``."""
    return f"{codigo}_seeds{ini:04d}-{fim - 1:04d}.npz"


def salvar_npz(caminho, dados):
    """Grava de forma atômica: um arquivo incompleto nunca fica com o nome final."""
    tmp = caminho + ".parcial.npz"
    np.savez_compressed(tmp, **dados)
    os.replace(tmp, caminho)


def pedaco_valido(caminho, seeds, horizonte):
    """O arquivo existe, abre e contém exatamente estas seeds neste horizonte?"""
    if not os.path.exists(caminho):
        return False
    try:
        with np.load(caminho) as z:
            return (np.array_equal(z["seeds"], np.array(seeds)) and int(z["horizonte"]) == horizonte
                    and all(k in z.files for k in CHAVES_POR_SEED))
    except Exception:
        return False


def lista_tarefas(cenarios, seed_inicio, seed_fim, tamanho_pedaco):
    """Lista de (cenário, ini, fim) em ordem fixa; o índice é o da opção --tarefa."""
    tarefas = []
    for codigo in cenarios:
        for ini in range(seed_inicio, seed_fim, tamanho_pedaco):
            tarefas.append((codigo, ini, min(ini + tamanho_pedaco, seed_fim)))
    return tarefas


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ensemble do modelo de vacinação (um .npz por pedaço de seeds).")
    ap.add_argument("--cenarios", nargs="+", default=["todos"],
                    help="códigos dos cenários (F0..F7, M0..M6) ou 'todos'")
    ap.add_argument("--seed-inicio", type=int, default=0, help="primeira seed (padrão 0)")
    ap.add_argument("--seed-fim", type=int, default=500, help="seed final, exclusiva (padrão 500)")
    ap.add_argument("--tamanho-pedaco", type=int, default=50, help="seeds por arquivo (padrão 50)")
    ap.add_argument("--processos", type=int, default=1, help="processos em paralelo (padrão 1)")
    ap.add_argument("--horizonte", type=int, default=1500, help="número de passos (padrão 1500)")
    ap.add_argument("--saida", default="results/pedacos", help="pasta dos pedaços")
    ap.add_argument("--tarefa", type=int, default=None,
                    help="roda só a tarefa com este índice (para job array); veja --listar")
    ap.add_argument("--listar", action="store_true", help="lista as tarefas e sai")
    ap.add_argument("--seeds-grades", type=int, nargs="*", default=[0],
                    help="seeds cujas grades de imunidade são guardadas (padrão: 0)")
    ap.add_argument("--refazer", action="store_true", help="roda de novo pedaços que já existem")
    a = ap.parse_args(argv)

    cenarios = list(CENARIOS) if a.cenarios == ["todos"] else a.cenarios
    for c in cenarios:
        if c not in CENARIOS:
            ap.error(f"cenário desconhecido: {c}")
    if a.seed_fim <= a.seed_inicio or a.tamanho_pedaco < 1 or a.processos < 1:
        ap.error("intervalo de seeds, tamanho de pedaço ou número de processos inválido")

    tarefas = lista_tarefas(cenarios, a.seed_inicio, a.seed_fim, a.tamanho_pedaco)
    if a.listar:
        for i, (c, ini, fim) in enumerate(tarefas):
            print(f"{i:4d}  {c}  seeds {ini}-{fim - 1}")
        print(f"{len(tarefas)} tarefas (índices 0 a {len(tarefas) - 1})")
        return 0
    if a.tarefa is not None:
        if not 0 <= a.tarefa < len(tarefas):
            ap.error(f"--tarefa deve estar entre 0 e {len(tarefas) - 1}")
        tarefas = [tarefas[a.tarefa]]

    os.makedirs(a.saida, exist_ok=True)
    executor = ProcessPoolExecutor(max_workers=a.processos) if a.processos > 1 else None
    try:
        for codigo, ini, fim in tarefas:
            seeds = list(range(ini, fim))
            caminho = os.path.join(a.saida, nome_pedaco(codigo, ini, fim))
            if not a.refazer and pedaco_valido(caminho, seeds, a.horizonte):
                print(f"[pula] {caminho} já existe", flush=True)
                continue
            t0 = time.perf_counter()
            dados = rodar_pedaco(codigo, seeds, a.horizonte, seeds_grades=tuple(a.seeds_grades),
                                 executor=executor)
            salvar_npz(caminho, dados)
            parede = time.perf_counter() - t0
            passos = int(np.maximum(dados["t_parada"], 1).sum())
            cpu = float(dados["tempo_s"].sum())
            print(f"[ok] {caminho}: {len(seeds)} seeds, {parede:.1f} s de parede, "
                  f"{cpu:.1f} s de CPU, {1e3 * cpu / passos:.2f} ms por passo simulado, "
                  f"{os.path.getsize(caminho) / 1e6:.2f} MB", flush=True)
    finally:
        if executor:
            executor.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
