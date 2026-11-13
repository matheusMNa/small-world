"""
Gera as figuras 05 a 12 do modelo de vacinação a partir de ``results/<código>.npz``.

Este script só lê os arquivos do ensemble; ele nunca roda simulações. Toda
figura com banda ou barra de erro sai em duas versões: ``_dp`` (média ±
desvio padrão entre seeds) e ``_ep`` (média ± erro padrão da média).
Grava também ``resumo_vacinacao.csv``, com as métricas de cada cenário na
simulação inteira e só na primeira onda. Faixas: 0 = nunca vacinado,
1 = m ≥ 0,90, 2 = 0,75 ≤ m < 0,90, 3 = 0,50 ≤ m < 0,75, 4 = m < 0,50.

Exemplo, a partir da raiz do repositório::

    python analyses/figuras_vacinacao.py
    python analyses/figuras_vacinacao.py --resultados /tmp/teste --saida /tmp/teste/plots --dpi 80
"""
import argparse
import csv
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")   # só grava arquivos; nunca abre janela
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model_vacinacao import CENARIOS, participacao_vacinados, primeira_onda, taxas_por_faixa

# Paleta do repositório (Okabe-Ito). A cor acompanha o tipo de cenário em todas as figuras.
COR_SEM_VACINA = '#D55E00'
COR_ALEATORIA = '#0072B2'
COR_GAUSSIANA = '#009E73'
COR_SEM_REFORCO = '#CC79A7'
# Faixas: nunca vacinado em preto; vacinados do mais para o menos protegido.
CORES_FAIXAS = ('#000000', '#0072B2', '#56B4E9', '#E69F00', '#D55E00')
ROTULOS_FAIXAS = ('Nunca vacinado', 'm ≥ 0,90', '0,75 ≤ m < 0,90', '0,50 ≤ m < 0,75', 'm < 0,50')

REGIMES = (("em t = 0", {"flu": ("F1", "F2"), "measles": ("M1", "M2")}),
           ("em t = 60", {"flu": ("F3", "F4"), "measles": ("M3", "M4")}),
           ("contínua (50 doses por passo)", {"flu": ("F5", "F6"), "measles": ("M5", "M6")}))
SEM_VACINA = {"flu": "F0", "measles": "M0"}
NOME_DOENCA = {"flu": "Flu-like", "measles": "Measles-like"}
NOME_BANDA = {"dp": "desvio padrão", "ep": "erro padrão"}


# ---------------------------------------------------------------------------
# Leitura e resumo
# ---------------------------------------------------------------------------
def estat(x, axis=0):
    """Média, desvio padrão amostral (ddof=1) e erro padrão ao longo das seeds, ignorando NaN."""
    x = np.asarray(x, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)   # fatias só com NaN ou n = 1
        n = np.sum(~np.isnan(x), axis=axis)
        media = np.nanmean(x, axis=axis)
        dp = np.nanstd(x, axis=axis, ddof=1)
        ep = dp / np.sqrt(np.maximum(n, 1))
    return dict(media=media, dp=dp, ep=ep, n=n)


def resumir_cenario(caminho):
    """Lê um ``results/<código>.npz`` e guarda só as estatísticas usadas nas figuras."""
    with np.load(caminho) as z:
        d = {k: z[k] for k in z.files}
    n_seeds = int(d["seeds"].size)
    horizonte = int(d["horizonte"])
    I = d["n_inf_faixa"].sum(axis=2)                       # (seeds, T+1)
    r = dict(n_seeds=n_seeds, horizonte=horizonte, I=estat(I))

    # primeira onda, definida sobre a média do ensemble
    t_pico, t_fim = primeira_onda(r["I"]["media"])
    r["onda"] = (t_pico, t_fim)

    # participação de vacinados nas infecções e nas transmissões
    for nome, fim in (("total", None), ("onda", t_fim)):
        f_inf, f_tr = participacao_vacinados(d, fim)
        r[f"part_inf_{nome}"] = estat(100 * f_inf)
        r[f"part_trans_{nome}"] = estat(100 * f_tr)

    # taxas por pessoa-passo em cada faixa (uma por seed, depois média entre seeds)
    lam_inf, lam_trans = taxas_por_faixa(d)
    r["lam_inf"], r["lam_trans"] = estat(lam_inf), estat(lam_trans)

    # tabela de métricas: simulação inteira e só a primeira onda
    linhas = []
    for janela, fim in (("total", None), ("primeira_onda", t_fim)):
        f_inf, f_tr = participacao_vacinados(d, fim)
        l_inf, l_tr = taxas_por_faixa(d, fim)
        metricas = {"infeccoes": d["novas_inf_faixa"][:, :fim].sum(axis=(1, 2)),
                    "pct_infeccoes_em_vacinados": 100 * f_inf,
                    "pct_transmissoes_por_vacinados": 100 * f_tr,
                    "primeiras_doses": d["doses1"][:, :fim].sum(axis=1),
                    "reforcos": d["reforcos"][:, :fim].sum(axis=1)}
        for b_ in range(5):
            metricas[f"taxa_infeccao_faixa{b_}"] = l_inf[:, b_]
            metricas[f"taxa_transmissao_faixa{b_}"] = l_tr[:, b_]
        for nome, valores in metricas.items():
            e = estat(valores)
            linhas.append((janela, nome, float(e["media"]), float(e["dp"]), float(e["ep"]), int(e["n"])))
    extintas = d["t_extincao"] >= 0
    linhas.append(("total", "fracao_execucoes_extintas", float(extintas.mean()), float("nan"), float("nan"), n_seeds))
    if extintas.any():
        e = estat(d["t_extincao"][extintas])
        linhas.append(("total", "passo_de_extincao", float(e["media"]), float(e["dp"]), float(e["ep"]), int(e["n"])))
    linhas.append(("total", "pico_da_primeira_onda_passo", float(t_pico), float("nan"), float("nan"), n_seeds))
    linhas.append(("total", "fim_da_primeira_onda_passo", float(t_fim), float("nan"), float("nan"), n_seeds))
    r["tabela"] = linhas

    # fração dos vacinados em cada faixa ao longo do tempo
    n_faixa = (d["n_inf_faixa"] + d["n_naoinf_faixa"])[:, :, 1:].astype(float)
    n_vac = n_faixa.sum(axis=2, keepdims=True)
    frac = np.divide(n_faixa, n_vac, out=np.full(n_faixa.shape, np.nan), where=n_vac > 0)
    r["frac_faixas"] = estat(frac)                          # (T+1, 4)
    r["reforcos"] = estat(d["reforcos"])
    r["doses1_total"] = estat(d["doses1"].sum(axis=1))

    # distribuição do número de infecções por indivíduo
    hist = d["hist_infeccoes"].astype(float)
    r["hist"] = estat(hist / hist.sum(axis=1, keepdims=True))

    r["extintas"] = float(np.mean(d["t_extincao"] >= 0))
    r["t_extincao"] = d["t_extincao"]
    r["grades"] = {k: d[k] for k in ("grades_seeds", "grades_tempos", "grades_m", "grades_infectado")}
    return r


def carregar(pasta):
    """Resumos de todos os cenários presentes em ``pasta``."""
    resumos = {}
    for codigo in CENARIOS:
        caminho = os.path.join(pasta, f"{codigo}.npz")
        if os.path.exists(caminho):
            resumos[codigo] = resumir_cenario(caminho)
        else:
            print(f"[aviso] {caminho} não existe; o cenário {codigo} fica fora das figuras")
    return resumos


def gravar_tabela(resumos, caminho):
    """Grava as métricas de todos os cenários em CSV (uma linha por cenário, janela e métrica)."""
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cenario", "n_seeds", "horizonte", "janela", "metrica", "media", "desvio_padrao",
                    "erro_padrao", "n_seeds_validas"])
        for codigo, r in resumos.items():
            for janela, nome, media, dp, ep, n in r["tabela"]:
                w.writerow([codigo, r["n_seeds"], r["horizonte"], janela, nome,
                            f"{media:.6g}", f"{dp:.6g}", f"{ep:.6g}", n])
    print(f"[ok] {caminho}")


def texto_seeds(resumos, codigos):
    """Texto 'n = 500 seeds' (ou o intervalo, se os cenários tiverem números diferentes)."""
    ns = sorted({resumos[c]["n_seeds"] for c in codigos if c in resumos})
    if not ns:
        return "n = 0 seeds"
    return f"n = {ns[0]} seeds" if len(ns) == 1 else f"n = {ns[0]} a {ns[-1]} seeds"


def salvar(fig, pasta, nome, dpi):
    caminho = os.path.join(pasta, nome)
    fig.savefig(caminho, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"[ok] {caminho}")


def curva_com_banda(ax, x, e, banda, cor, rotulo, estilo='-'):
    """Linha da média com a banda (dp ou ep) sombreada."""
    media, larg = e["media"], np.nan_to_num(e[banda])
    ax.fill_between(x, media - larg, media + larg, color=cor, alpha=0.22, linewidth=0)
    ax.plot(x, media, estilo, color=cor, linewidth=2, label=rotulo)


def estilo_eixo(ax):
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.spines[['top', 'right']].set_visible(False)


def curvas_do_painel(doenca, par):
    """(código, cor, rótulo, estilo) das curvas de um painel de regime."""
    return [(SEM_VACINA[doenca], COR_SEM_VACINA, 'Sem vacina', '-'),
            (par[0], COR_ALEATORIA, 'Alocação aleatória', '-'),
            (par[1], COR_GAUSSIANA, 'Alocação gaussiana (grupos de risco)', '-')]


# ---------------------------------------------------------------------------
# Figuras 05, 06 e 07: infectados ao longo do tempo
# ---------------------------------------------------------------------------
def fig_infectados(resumos, doenca, banda, primeira=False):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    usados, x_max = [], 0
    for ax, (nome_regime, pares) in zip(axes, REGIMES):
        curvas = curvas_do_painel(doenca, pares[doenca])
        if doenca == "flu" and pares is REGIMES[0][1]:
            curvas.append(("F7", COR_SEM_REFORCO, 'Aleatória, sem reforço (controle)', '--'))
        fim_painel = 0
        for codigo, cor, rotulo, estilo in curvas:
            if codigo not in resumos:
                continue
            r = resumos[codigo]
            usados.append(codigo)
            x = np.arange(r["horizonte"] + 1)
            curva_com_banda(ax, x, r["I"], banda, cor, rotulo, estilo)
            if primeira:
                ax.axvline(r["onda"][1], color=cor, linestyle=':', linewidth=1.5)
                fim_painel = max(fim_painel, r["onda"][1])
            else:
                ativos = np.flatnonzero(r["I"]["media"] > 0.5)
                fim_painel = max(fim_painel, ativos[-1] if ativos.size else 0)
        ax.set_title(f"Vacinação {nome_regime}", fontsize=13)
        ax.set_xlabel('Tempo (passos)', fontsize=12)
        estilo_eixo(ax)
        x_max = max(x_max, fim_painel)
        if primeira and fim_painel:
            ax.set_xlim(0, fim_painel * 1.05)
    if not usados:
        plt.close(fig)
        return None
    horizonte = max(resumos[c]["horizonte"] for c in usados)
    if not primeira:
        # No Measles-like a epidemia acaba cedo; corta o eixo depois do último infectado.
        limite = horizonte if doenca == "flu" else min(horizonte, int(x_max * 1.1) + 10)
        for ax in axes:
            ax.set_xlim(0, limite)
    axes[0].set_ylabel('Indivíduos infectados', fontsize=12)
    axes[0].set_ylim(bottom=0)

    itens = [Line2D([], [], color=COR_SEM_VACINA, lw=2, label='Sem vacina'),
             Line2D([], [], color=COR_ALEATORIA, lw=2, label='Alocação aleatória'),
             Line2D([], [], color=COR_GAUSSIANA, lw=2, label='Alocação gaussiana (grupos de risco)')]
    if doenca == "flu":
        itens.append(Line2D([], [], color=COR_SEM_REFORCO, lw=2, ls='--',
                            label='Aleatória, sem reforço (só em t = 0)'))
    if primeira:
        itens.append(Line2D([], [], color='gray', lw=1.5, ls=':', label='Fim da primeira onda de cada cenário'))
    fig.legend(handles=itens, loc='lower center', ncol=len(itens), fontsize=11, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    oque = "primeira onda de infectados" if primeira else "infectados ao longo do tempo"
    corte = ""
    if doenca == "measles" and not primeira and limite < horizonte:
        corte = f"; eixo cortado em t = {limite}, depois disso não há infectados"
    fig.suptitle(f"{NOME_DOENCA[doenca]}: {oque} (média ± {NOME_BANDA[banda]}, "
                 f"{texto_seeds(resumos, usados)}{corte})", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figura 08: participação dos vacinados
# ---------------------------------------------------------------------------
def fig_participacao(resumos, banda):
    codigos = [c for c in CENARIOS if c in resumos and CENARIOS[c].get("regime") != "nenhum"]
    if not codigos:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.5), sharey=True)
    x = np.arange(len(codigos))
    larg = 0.38
    for ax, chave, titulo in ((axes[0], "part_inf", "Infecções sofridas por vacinados"),
                              (axes[1], "part_trans", "Transmissões feitas por vacinados")):
        tot = [float(resumos[c][f"{chave}_total"]["media"]) for c in codigos]
        tot_e = [float(np.nan_to_num(resumos[c][f"{chave}_total"][banda])) for c in codigos]
        ax.bar(x - larg / 2, tot, larg, yerr=tot_e, color=COR_ALEATORIA, edgecolor='white',
               capsize=3, label='Simulação inteira')
        # a primeira onda só é mostrada no Flu-like; no Measles-like há uma onda só
        flu = [i for i, c in enumerate(codigos) if c.startswith("F")]
        if flu:
            onda = [float(resumos[codigos[i]][f"{chave}_onda"]["media"]) for i in flu]
            onda_e = [float(np.nan_to_num(resumos[codigos[i]][f"{chave}_onda"][banda])) for i in flu]
            ax.bar(x[flu] + larg / 2, onda, larg, yerr=onda_e, color='#56B4E9', edgecolor='white',
                   hatch='//', capsize=3, label='Só a primeira onda (Flu-like)')
        ax.set_xticks(x)
        ax.set_xticklabels(codigos, fontsize=11)
        ax.set_title(titulo, fontsize=13)
        ax.set_xlabel('Cenário', fontsize=12)
        ax.set_ylim(0, 100)
        estilo_eixo(ax)
    axes[0].set_ylabel('% do total de eventos', fontsize=12)
    axes[0].legend(fontsize=11, frameon=False, loc='upper left')
    fig.suptitle("Participação dos vacinados nas infecções e nas transmissões "
                 f"(média ± {NOME_BANDA[banda]}, {texto_seeds(resumos, codigos)})", fontsize=14, y=1.0)
    fig.text(0.5, -0.03, "Vacinado = recebeu ao menos uma dose até o instante do evento. A porcentagem depende de "
             "quantas pessoas estão vacinadas; para comparar riscos, veja as taxas por faixa (figura 09).",
             ha='center', fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figura 09: taxas por pessoa-passo em cada faixa
# ---------------------------------------------------------------------------
def fig_taxas(resumos, banda):
    grupos = {"flu": [c for c in CENARIOS if c.startswith("F") and c in resumos],
              "measles": [c for c in CENARIOS if c.startswith("M") and c in resumos]}
    if not any(grupos.values()):
        return None
    fig, axes = plt.subplots(2, 2, figsize=(17, 9), sharey='row')
    larg = 0.16
    for j, doenca in enumerate(("flu", "measles")):
        codigos = grupos[doenca]
        x = np.arange(len(codigos))
        for i, (chave, nome) in enumerate((("lam_inf", "Infecções por pessoa-passo exposta"),
                                           ("lam_trans", "Transmissões por pessoa-passo infectada"))):
            ax = axes[i, j]
            for b in range(5):
                val = np.array([resumos[c][chave]["media"][b] for c in codigos], dtype=float)
                err = np.array([np.nan_to_num(resumos[c][chave][banda][b]) for c in codigos], dtype=float)
                ok = np.isfinite(val) & (val > 0)
                if not ok.any():
                    continue
                # no eixo logarítmico a barra de erro não pode chegar a zero
                baixo = np.minimum(err[ok], 0.9 * val[ok])
                ax.bar(x[ok] + (b - 2) * larg, val[ok], larg, yerr=[baixo, err[ok]], color=CORES_FAIXAS[b],
                       edgecolor='white', hatch='//' if b == 0 else None, capsize=2,
                       error_kw=dict(ecolor='#555555', lw=1))
            ax.set_yscale('log')
            ax.set_xticks(x)
            ax.set_xticklabels(codigos, fontsize=11)
            ax.set_title(f"{NOME_DOENCA[doenca]}: {nome.lower()}", fontsize=13)
            estilo_eixo(ax)
            if j == 0:
                ax.set_ylabel(f"{nome}\n(escala logarítmica)", fontsize=11)
            if i == 1:
                ax.set_xlabel('Cenário', fontsize=12)
    itens = [Patch(facecolor=CORES_FAIXAS[b], edgecolor='white', hatch='//' if b == 0 else None,
                   label=ROTULOS_FAIXAS[b]) for b in range(5)]
    fig.legend(handles=itens, loc='lower center', ncol=5, fontsize=11, frameon=False,
               bbox_to_anchor=(0.5, -0.04), title='Faixa de imunidade no instante do evento')
    todos = grupos["flu"] + grupos["measles"]
    fig.suptitle("Taxas por pessoa-passo em cada faixa de imunidade "
                 f"(média ± {NOME_BANDA[banda]}, {texto_seeds(resumos, todos)})", fontsize=14, y=1.0)
    fig.text(0.5, -0.075, "Só contam passos com ao menos um infectado. Faixas sem pessoas-passo ou sem eventos "
             "não aparecem.", ha='center', fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figura 10: faixas dos vacinados e reforços ao longo do tempo (Flu-like)
# ---------------------------------------------------------------------------
def fig_faixas_reforcos(resumos, banda):
    usados = [c for c in ("F1", "F2", "F3", "F4", "F5", "F6") if c in resumos]
    if not usados:
        return None
    fig, axes = plt.subplots(3, 3, figsize=(17, 11), sharex=True, sharey='row')
    for j, (nome_regime, pares) in enumerate(REGIMES):
        for i, (codigo, nome_aloc) in enumerate(zip(pares["flu"], ("aleatória", "gaussiana"))):
            ax = axes[i, j]
            ax.set_title(f"{codigo}: vacinação {nome_regime}, {nome_aloc}", fontsize=12)
            estilo_eixo(ax)
            if codigo not in resumos:
                continue
            r = resumos[codigo]
            x = np.arange(r["horizonte"] + 1)
            for b in range(4):
                e = {k: r["frac_faixas"][k][:, b] for k in ("media", "dp", "ep")}
                curva_com_banda(ax, x, e, banda, CORES_FAIXAS[b + 1], ROTULOS_FAIXAS[b + 1])
            ax.set_ylim(0, 1)
        ax = axes[2, j]
        estilo_eixo(ax)
        ax.set_title(f"Reforços por passo, vacinação {nome_regime}", fontsize=12)
        for codigo, cor, rotulo in ((pares["flu"][0], COR_ALEATORIA, 'Alocação aleatória'),
                                    (pares["flu"][1], COR_GAUSSIANA, 'Alocação gaussiana')):
            if codigo in resumos:
                r = resumos[codigo]
                curva_com_banda(ax, np.arange(r["horizonte"] + 1), r["reforcos"], banda, cor, rotulo)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Tempo (passos)', fontsize=12)
    axes[0, 0].set_ylabel('Fração dos vacinados\n(alocação aleatória)', fontsize=11)
    axes[1, 0].set_ylabel('Fração dos vacinados\n(alocação gaussiana)', fontsize=11)
    axes[2, 0].set_ylabel('Doses de reforço por passo', fontsize=11)
    itens = [Line2D([], [], color=CORES_FAIXAS[b], lw=2, label=f"Faixa {ROTULOS_FAIXAS[b]}") for b in range(1, 5)]
    itens += [Line2D([], [], color=COR_ALEATORIA, lw=2, label='Reforços, alocação aleatória'),
              Line2D([], [], color=COR_GAUSSIANA, lw=2, label='Reforços, alocação gaussiana')]
    fig.legend(handles=itens, loc='lower center', ncol=6, fontsize=10.5, frameon=False,
               bbox_to_anchor=(0.5, -0.035))
    fig.suptitle("Flu-like: distribuição dos vacinados pelas faixas de imunidade e doses de reforço "
                 f"(média ± {NOME_BANDA[banda]}, {texto_seeds(resumos, usados)})", fontsize=14, y=1.0)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figura 11: infecções por indivíduo (Flu-like)
# ---------------------------------------------------------------------------
def fig_infeccoes_por_individuo(resumos):
    usados = [c for c in CENARIOS if c.startswith("F") and c in resumos]
    if not usados:
        return None
    k_max = 1
    for c in usados:
        visiveis = np.flatnonzero(resumos[c]["hist"]["media"] > 1e-4)
        k_max = max(k_max, visiveis[-1] if visiveis.size else 1)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    for ax, (nome_regime, pares) in zip(axes, REGIMES):
        curvas = curvas_do_painel("flu", pares["flu"])
        if pares is REGIMES[0][1]:
            curvas.append(("F7", COR_SEM_REFORCO, 'Aleatória, sem reforço (controle)', '--'))
        for codigo, cor, rotulo, estilo in curvas:
            if codigo in resumos:
                h = resumos[codigo]["hist"]["media"][:k_max + 1]
                ax.plot(np.arange(k_max + 1), 100 * h, estilo, marker='o', markersize=5, color=cor,
                        linewidth=2, label=rotulo)
        ax.set_title(f"Vacinação {nome_regime}", fontsize=13)
        ax.set_xlabel('Número de infecções sofridas pelo indivíduo', fontsize=12)
        estilo_eixo(ax)
    axes[0].set_ylabel('% dos indivíduos (média entre seeds)', fontsize=12)
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=10.5, frameon=False)
    horizonte = max(resumos[c]["horizonte"] for c in usados)
    fig.suptitle(f"Flu-like: distribuição do número de infecções por indivíduo em {horizonte} passos "
                 f"({texto_seeds(resumos, usados)})", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figura 12: grades de imunidade
# ---------------------------------------------------------------------------
def fig_grades(resumos, seed=0):
    usados = [c for c in ("F1", "F2") if c in resumos and seed in resumos[c]["grades"]["grades_seeds"]]
    if not usados:
        return None
    tempos = resumos[usados[0]]["grades"]["grades_tempos"]
    fig, axes = plt.subplots(len(usados), len(tempos), figsize=(4.1 * len(tempos), 4.3 * len(usados)),
                             squeeze=False)
    cmap_inf = ListedColormap([(0, 0, 0, 0), COR_SEM_VACINA])
    nomes = {"F1": "F1: alocação aleatória", "F2": "F2: alocação gaussiana"}
    for i, codigo in enumerate(usados):
        g = resumos[codigo]["grades"]
        k = int(np.flatnonzero(g["grades_seeds"] == seed)[0])
        for j, t in enumerate(tempos):
            ax = axes[i, j]
            im = ax.imshow(g["grades_m"][k, j], cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
            ax.imshow(g["grades_infectado"][k, j].astype(int), cmap=cmap_inf, vmin=0, vmax=1,
                      interpolation='nearest')
            ax.set_title(f"t = {int(t)}", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(nomes[codigo], fontsize=12)
    barra = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    barra.set_label('Imunidade m', fontsize=12)
    fig.legend(handles=[Patch(facecolor=COR_SEM_VACINA, label='Infectado')], loc='lower center',
               fontsize=11, frameon=False, bbox_to_anchor=(0.45, 0.02))
    fig.suptitle(f"Flu-like, vacinação em t = 0: mapas de imunidade de uma única execução (seed {seed})",
                 fontsize=14, y=0.98)
    return fig


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Figuras do modelo de vacinação a partir de results/<código>.npz.")
    ap.add_argument("--resultados", default="results", help="pasta com os arquivos <código>.npz")
    ap.add_argument("--saida", default="analyses/plots", help="pasta das figuras")
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args(argv)

    resumos = carregar(a.resultados)
    if not resumos:
        print(f"Nenhum resultado em {a.resultados}. Rode analyses/ensemble.py e analyses/juntar_pedacos.py antes.",
              file=sys.stderr)
        return 1
    os.makedirs(a.saida, exist_ok=True)

    for codigo, r in resumos.items():
        ext = r["t_extincao"][r["t_extincao"] >= 0]
        quando = f", passo mediano de extinção {int(np.median(ext))}" if ext.size else ""
        print(f"{codigo}: {r['n_seeds']} seeds, horizonte {r['horizonte']}, primeiras doses em média "
              f"{float(r['doses1_total']['media']):.0f}, primeira onda até t = {r['onda'][1]}, "
              f"{100 * r['extintas']:.0f}% das execuções extintas{quando}")

    def gravar(fig, nome):
        if fig is None:
            print(f"[aviso] sem dados para {nome}")
        else:
            salvar(fig, a.saida, nome, a.dpi)

    for banda in ("dp", "ep"):
        gravar(fig_infectados(resumos, "flu", banda), f"05_flu_infectados_{banda}.png")
        gravar(fig_infectados(resumos, "measles", banda), f"06_measles_infectados_{banda}.png")
        gravar(fig_infectados(resumos, "flu", banda, primeira=True), f"07_flu_primeira_onda_{banda}.png")
        gravar(fig_participacao(resumos, banda), f"08_participacao_vacinados_{banda}.png")
        gravar(fig_taxas(resumos, banda), f"09_taxas_por_faixa_{banda}.png")
        gravar(fig_faixas_reforcos(resumos, banda), f"10_flu_faixas_e_reforcos_{banda}.png")
    gravar(fig_infeccoes_por_individuo(resumos), "11_flu_infeccoes_por_individuo.png")
    gravar(fig_grades(resumos), "12_grades_imunidade.png")
    gravar_tabela(resumos, os.path.join(a.saida, "resumo_vacinacao.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
