"""
Reúne os pedaços gerados por ``analyses/ensemble.py`` em um arquivo por cenário.

Para cada cenário, lê ``<pedacos>/<código>_seeds*.npz``, confere que as seeds
0 a n-1 estão todas presentes, sem repetição e com os mesmos parâmetros, e
grava ``<saida>/<código>.npz``. Se faltar ou sobrar alguma seed, o cenário não
é gravado e o script termina com erro, dizendo quais seeds são o problema.

Exemplo, a partir da raiz do repositório::

    python analyses/juntar_pedacos.py                      # 500 seeds, todos os cenários
    python analyses/juntar_pedacos.py --n-seeds 3 --pedacos /tmp/teste/pedacos --saida /tmp/teste
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from analyses.ensemble import CHAVES_GRADES, CHAVES_POR_SEED, salvar_npz
from src.model_vacinacao import CENARIOS


def resumir(numeros, limite=12):
    """Lista curta de números para mensagens de erro."""
    numeros = sorted(int(n) for n in numeros)
    texto = ", ".join(str(n) for n in numeros[:limite])
    return texto + (f", ... ({len(numeros)} no total)" if len(numeros) > limite else "")


def juntar_cenario(codigo, pasta_pedacos, n_seeds):
    """Devolve os dados reunidos do cenário ou levanta ValueError com o motivo."""
    arquivos = sorted(glob.glob(os.path.join(pasta_pedacos, f"{codigo}_seeds*.npz")))
    arquivos = [f for f in arquivos if not f.endswith(".parcial.npz")]
    if not arquivos:
        raise ValueError(f"nenhum pedaço encontrado em {pasta_pedacos}")

    partes = []
    for f in arquivos:
        try:
            with np.load(f) as z:
                partes.append({k: z[k] for k in z.files})
        except Exception as e:
            raise ValueError(f"não foi possível ler {f}: {e}")

    ref = partes[0]
    for f, parte in zip(arquivos, partes):
        if str(parte["parametros_json"]) != str(ref["parametros_json"]):
            raise ValueError(f"{f} foi gerado com parâmetros diferentes dos de {arquivos[0]}")
        if not np.array_equal(parte["grades_tempos"], ref["grades_tempos"]):
            raise ValueError(f"{f} guarda grades em instantes diferentes dos de {arquivos[0]}")

    seeds = np.concatenate([parte["seeds"] for parte in partes])
    valores, contagens = np.unique(seeds, return_counts=True)
    repetidas = valores[contagens > 1]
    faltando = np.setdiff1d(np.arange(n_seeds), valores)
    sobrando = np.setdiff1d(valores, np.arange(n_seeds))
    problemas = []
    if repetidas.size:
        problemas.append(f"seeds repetidas: {resumir(repetidas)}")
    if faltando.size:
        problemas.append(f"seeds faltando: {resumir(faltando)}")
    if sobrando.size:
        problemas.append(f"seeds fora de 0..{n_seeds - 1}: {resumir(sobrando)}")
    if problemas:
        raise ValueError("; ".join(problemas))

    ordem = np.argsort(seeds, kind="stable")
    dados = {k: np.concatenate([parte[k] for parte in partes])[ordem] for k in CHAVES_POR_SEED}
    dados["seeds"] = seeds[ordem].astype(np.int32)

    seeds_g = np.concatenate([parte["grades_seeds"] for parte in partes])
    ordem_g = np.argsort(seeds_g, kind="stable")
    dados["grades_seeds"] = seeds_g[ordem_g].astype(np.int32)
    for k in CHAVES_GRADES:
        dados[k] = np.concatenate([parte[k] for parte in partes])[ordem_g]
    for k in ("grades_tempos", "codigo", "horizonte", "parametros_json"):
        dados[k] = ref[k]
    return dados


def main(argv=None):
    ap = argparse.ArgumentParser(description="Reúne os pedaços do ensemble em results/<código>.npz.")
    ap.add_argument("--pedacos", default="results/pedacos", help="pasta com os pedaços")
    ap.add_argument("--saida", default="results", help="pasta dos arquivos reunidos")
    ap.add_argument("--cenarios", nargs="+", default=["todos"])
    ap.add_argument("--n-seeds", type=int, default=500,
                    help="número de seeds esperado; exige exatamente as seeds 0..n-1 (padrão 500)")
    a = ap.parse_args(argv)

    cenarios = list(CENARIOS) if a.cenarios == ["todos"] else a.cenarios
    os.makedirs(a.saida, exist_ok=True)
    falhas = 0
    for codigo in cenarios:
        try:
            dados = juntar_cenario(codigo, a.pedacos, a.n_seeds)
        except ValueError as e:
            falhas += 1
            print(f"[ERRO] {codigo}: {e}", file=sys.stderr)
            continue
        caminho = os.path.join(a.saida, f"{codigo}.npz")
        salvar_npz(caminho, dados)
        print(f"[ok] {caminho}: {dados['seeds'].size} seeds, horizonte {int(dados['horizonte'])}, "
              f"{os.path.getsize(caminho) / 1e6:.1f} MB")
    if falhas:
        print(f"{falhas} cenário(s) com erro; nada foi gravado para eles.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
