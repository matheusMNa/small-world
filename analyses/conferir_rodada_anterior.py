"""
Confere que a rodada nova, com bloco central, reproduz a rodada anterior.

A rodada anterior gravou ``<anterior>/F0.npz`` ... ``<anterior>/M6.npz`` (15
cenários, só bloco central). A rodada nova grava os mesmos cenários como
``<nova>/F0C.npz`` ... ``<nova>/M6C.npz``. Como a dinâmica com bloco central
não mudou, as séries que já existiam têm de ser idênticas, seed a seed.

Exemplo, a partir da raiz do repositório::

    python analyses/conferir_rodada_anterior.py --anterior results_v1 --nova results
"""
import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model_vacinacao import CENARIOS

# Grandezas gravadas pelas duas versões, com uma entrada por seed.
CHAVES_COMUNS = ("n_inf_faixa", "n_naoinf_faixa", "novas_inf_faixa", "trans_faixa", "doses1", "reforcos",
                 "imunidade_media", "hist_infeccoes", "t_extincao", "t_parada")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compara <anterior>/<cenário>.npz com <nova>/<cenário>C.npz.")
    ap.add_argument("--anterior", default="results_v1", help="pasta da rodada anterior (padrão results_v1)")
    ap.add_argument("--nova", default="results", help="pasta da rodada nova (padrão results)")
    a = ap.parse_args(argv)

    diferentes = ausentes = 0
    for base in CENARIOS:
        f_ant = os.path.join(a.anterior, f"{base}.npz")
        f_nov = os.path.join(a.nova, f"{base}C.npz")
        if not (os.path.exists(f_ant) and os.path.exists(f_nov)):
            ausentes += 1
            print(f"[falta] {base}: é preciso ter {f_ant} e {f_nov}")
            continue
        with np.load(f_ant) as z_ant, np.load(f_nov) as z_nov:
            seeds = np.intersect1d(z_ant["seeds"], z_nov["seeds"])
            if int(z_ant["horizonte"]) != int(z_nov["horizonte"]) or seeds.size == 0:
                ausentes += 1
                print(f"[falta] {base}: horizontes diferentes ou nenhuma seed em comum")
                continue
            i_ant = np.searchsorted(z_ant["seeds"], seeds)
            i_nov = np.searchsorted(z_nov["seeds"], seeds)
            ruins = [k for k in CHAVES_COMUNS if not np.array_equal(z_ant[k][i_ant], z_nov[k][i_nov])]
        if ruins:
            diferentes += 1
            print(f"[DIFERENTE] {base}: {', '.join(ruins)} ({seeds.size} seeds comparadas)")
        else:
            print(f"[idêntico] {base} e {base}C: {len(CHAVES_COMUNS)} grandezas, {seeds.size} seeds")
    if diferentes:
        print(f"\n{diferentes} cenário(s) diferem da rodada anterior. Confira as versões do Python e do NumPy "
              "usadas nas duas rodadas.", file=sys.stderr)
    elif ausentes:
        print(f"\n{ausentes} cenário(s) não puderam ser comparados.", file=sys.stderr)
    else:
        print("\nA rodada nova com bloco central é idêntica à anterior.")
    return 1 if (diferentes or ausentes) else 0


if __name__ == "__main__":
    sys.exit(main())
