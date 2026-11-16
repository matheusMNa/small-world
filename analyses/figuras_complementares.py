"""
Figuras 13 a 19 do modelo de vacinação, a partir dos resumos de ``figuras_vacinacao.carregar``.

* 13: evolução espacial em quatro estados (suscetível, infectado, recuperado, vacinado);
* 14: comparação do ensemble sem vacina com as EDOs de campo médio (SIR e SIRS);
* 15: efeito da semente inicial (número de infectados iniciais, em bloco ou espalhados);
* 16: efeito das conexões globais (K encontros de mercado por passo);
* 17: regimes de vacinação: preventiva (t = 0), reativa (t = 60) e contínua;
* 18: percolação dos suscetíveis (parâmetro de ordem e mapas de aglomerados);
* 19: comparação entre infectados iniciais em bloco central (C) e espalhados (E).

Este módulo não roda simulações e não tem linha de comando própria: as figuras
são gravadas por ``python analyses/figuras_vacinacao.py`` e podem ser desenhadas
uma a uma no notebook ``analyses/figuras_vacinacao.ipynb``.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.integrate import odeint
from scipy.optimize import brentq

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import figuras_vacinacao as FV
from src.model_vacinacao import (CENARIOS, INFECTADO, K_VARREDURA, LADOS_SEMENTE, N_INICIAIS, NOMES_ESTADOS,
                                 P_C_MOORE, SUSCETIVEL, aglomerados_periodicos, medidas_percolacao)

# Estados: as cores de S, I e R são as das figuras 00 a 04 do modelo original.
CORES_ESTADOS = ('#56B4E9', '#D55E00', '#009E73', '#AA4499')
# Regimes de vacinação (figura 17): a cor é o regime; o traço é a alocação.
COR_REGIME = {"nenhum": '#D55E00', "t0": '#0072B2', "t60": '#CC79A7', "continuo": '#E69F00'}
NOME_REGIME = {"nenhum": "Sem vacina", "t0": "Preventiva (t = 0)", "t60": "Reativa (t = 60)",
               "continuo": "Contínua (50 doses por passo)"}
# Grandeza ordenada (I0 ou K): um só matiz, do claro para o escuro.
RAMPA = ('#9ECAE1', '#4292C6', '#2171B5', '#08306B')
COR_INERTE = '#E0E0E0'
PREFIXO = {"flu": "F", "measles": "M"}
BANDAS = ("dp", "ep")


def regime_do_cenario(base):
    """"nenhum", "t0", "t60" ou "continuo"."""
    c = CENARIOS[base]
    if c.get("regime") == "nenhum":
        return "nenhum"
    if c.get("regime") == "continuo":
        return "continuo"
    return "t0" if c.get("t_vac", 0) == 0 else "t60"


def descricao(base):
    """Texto curto do cenário, por exemplo "F2: t = 0, gaussiana"."""
    c = CENARIOS[base]
    reg = regime_do_cenario(base)
    if reg == "nenhum":
        return f"{base}: sem vacina"
    quando = {"t0": "t = 0", "t60": "t = 60", "continuo": "contínua"}[reg]
    extra = ", sem reforço" if c.get("reforco") is False else ""
    return f"{base}: {quando}, {c['alocacao']}{extra}"


def bases(doenca, resumos_cond):
    return [b for b in CENARIOS if b.startswith(PREFIXO[doenca]) and b in resumos_cond]


def fim_da_epidemia(lista_resumos, folga=1.1):
    """Último passo em que a média de infectados passa de 0,5 em algum dos resumos, com folga."""
    fim = 0
    for r in lista_resumos:
        ativos = np.flatnonzero(r["I"]["media"] > 0.5)
        fim = max(fim, int(ativos[-1]) if ativos.size else 0)
    horizonte = max(r["horizonte"] for r in lista_resumos)
    return min(horizonte, int(fim * folga) + 10)


def seed_representativa(r):
    """Posição, entre as seeds com grades, daquela cujo tamanho final é o mais próximo da mediana.

    Evita mostrar uma execução atípica (por exemplo, um surto que se extinguiu
    cedo quando a maioria não se extingue). Devolve ``(posição, seed)``.
    """
    gs = r["grades"]["grades_seeds"]
    if gs.size == 0:
        return None, None
    pos = np.searchsorted(r["seeds"], gs)
    mediana = np.median(r["final_seeds"])
    k = int(np.argmin(np.abs(r["final_seeds"][pos] - mediana)))
    return k, int(gs[k])


def legenda_estados(fig, y=0.0, n=4):
    itens = [Patch(facecolor=CORES_ESTADOS[e], edgecolor='none', label=NOMES_ESTADOS[e]) for e in range(n)]
    fig.legend(handles=itens, loc='lower center', ncol=n, fontsize=11, frameon=False, bbox_to_anchor=(0.5, y))


# ---------------------------------------------------------------------------
# Figura 13: evolução espacial em quatro estados
# ---------------------------------------------------------------------------
def fig_evolucao_espacial(resumos, doenca, cond, tempos=(0, 50, 100, 150, 300, 600, None)):
    """Uma linha por cenário, uma coluna por instante; ``None`` em ``tempos`` é o horizonte."""
    fam = FV.por_condicao(resumos, cond)
    usados = [b for b in bases(doenca, fam) if "grades_estado" in fam[b]["grades"]
              and fam[b]["grades"]["grades_seeds"].size]
    if not usados:
        return None
    horizonte = fam[usados[0]]["horizonte"]
    guardados = [int(t) for t in fam[usados[0]]["grades"]["grades_tempos"]]
    tempos = [horizonte if t is None else int(t) for t in tempos]
    tempos = [t for t in dict.fromkeys(tempos) if t in guardados]
    cmap = ListedColormap(CORES_ESTADOS)
    fig, axes = plt.subplots(len(usados), len(tempos), figsize=(2.25 * len(tempos) + 0.6, 2.35 * len(usados) + 1.1),
                             squeeze=False)
    for i, base in enumerate(usados):
        r = fam[base]
        k, seed = seed_representativa(r)
        for j, t in enumerate(tempos):
            ax = axes[i, j]
            ax.imshow(r["grades"]["grades_estado"][k, guardados.index(t)], cmap=cmap, vmin=-0.5, vmax=3.5,
                      interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"t = {t}", fontsize=12)
            if j == 0:
                ax.set_ylabel(f"{descricao(base)}\n(seed {seed})", fontsize=9.5)
    fig.suptitle(f"{FV.NOME_DOENCA[doenca]}: evolução espacial de uma execução por cenário "
                 f"(infectados iniciais: {FV.NOME_CONDICAO[cond]})", fontsize=14, y=1.0)
    fig.text(0.5, 0.965, "Em cada cenário, a seed mostrada é a de tamanho final mais próximo da mediana do ensemble. "
             "Suscetível = não infectado com imunidade m < 0,5.", ha='center', va='top', fontsize=10)
    fig.tight_layout(rect=(0, 0.035, 1, 0.955))
    legenda_estados(fig, y=0.0)
    return fig


# ---------------------------------------------------------------------------
# Figura 14: comparação com as EDOs de campo médio
# ---------------------------------------------------------------------------
def taxas_campo_medio(par):
    """(beta, gamma, omega) das EDOs de campo médio que correspondem aos parâmetros ``par``.

    * ``beta = alpha (8 W_local + K W_mercado) <S0> <T0>``: mistura homogênea, com
      S0 e T0 independentes e substituídos pelas médias, e 1 - exp(-x) ~ x;
    * ``gamma = p_rec``;
    * ``omega = 1 / (tau_n ln 2)``: inverso do tempo que a imunidade natural leva
      para cair abaixo de 0,5 (zero quando a imunidade é permanente). No modelo
      esse tempo é fixo e a proteção cai aos poucos; a EDO troca isso por uma
      perda com taxa constante, então a comparação no Flu-like é só aproximada.
    """
    media = 0.5 * (par["atrib_min"] + par["atrib_max"])
    beta = par["alpha"] * (8 * par["W_local"] + par["K_encontros"] * par["W_mercado"]) * media ** 2
    tau_n = par["tau_n"]
    omega = 0.0 if np.isinf(tau_n) else 1.0 / (tau_n * np.log(2))
    return beta, par["p_rec"], omega


def infectados_iniciais(par):
    return par["init_espalhados"] if par.get("init_espalhados", 0) > 0 else par["init_cluster"] ** 2


def solucao_edo(par, t):
    """S(t), I(t), R(t) da EDO SIR (omega = 0) ou SIRS, com os mesmos N e I(0) da simulação."""
    beta, gamma, omega = taxas_campo_medio(par)
    N = par["N"] ** 2
    i0 = infectados_iniciais(par)

    def derivada(y, _t):
        S, I, R = y
        novas = beta * S * I / N
        return [-novas + omega * R, novas - gamma * I, gamma * I - omega * R]

    return odeint(derivada, [N - i0, i0, 0.0], np.asarray(t, dtype=float)).T


def fig_edo(resumos, banda, t_max_flu=600):
    """Ensemble sem vacina (linhas cheias, com banda) e EDO de campo médio (tracejado)."""
    paineis = [(doenca, cond, f"{PREFIXO[doenca]}0{cond}") for doenca in ("measles", "flu") for cond in ("C", "E")]
    paineis = [x for x in paineis if x[2] in resumos and "estado" in resumos[x[2]]]
    if not paineis:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(17, 9.5), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for doenca, cond, codigo in paineis:
        ax = axes[0 if doenca == "measles" else 1, 0 if cond == "C" else 1]
        ax.set_visible(True)
        r = resumos[codigo]
        t = np.arange(r["horizonte"] + 1)
        edo = solucao_edo(r["param"], t)
        for e in (0, 1, 2):
            est = {k: r["estado"][k][:, e] for k in ("media", "dp", "ep")}
            FV.curva_com_banda(ax, t, est, banda, CORES_ESTADOS[e], None)
            ax.plot(t, edo[e], '--', color=CORES_ESTADOS[e], linewidth=1.6)
        beta, gamma, omega = taxas_campo_medio(r["param"])
        modelo = "SIR" if omega == 0 else "SIRS"
        ax.set_xlim(0, fim_da_epidemia([r], folga=1.3) if doenca == "measles" else min(t_max_flu, r["horizonte"]))
        ax.set_ylim(0, r["pop"] * 1.02)
        ax.set_title(f"{codigo}: {FV.NOME_DOENCA[doenca]} sem vacina, infectados iniciais {FV.NOME_CONDICAO[cond]}\n"
                     f"EDO {modelo}: β = {beta:.3f}, γ = {gamma:.2f}"
                     + (f", ω = {omega:.4f}" if omega else "") + f", R₀ = β/γ = {beta / gamma:.1f}", fontsize=11.5)
        ax.set_xlabel('Tempo (passos)', fontsize=12)
        ax.set_ylabel('Indivíduos', fontsize=12)
        FV.estilo_eixo(ax)
    itens = [Line2D([], [], color=CORES_ESTADOS[e], lw=2, label=f"{NOMES_ESTADOS[e]} (ensemble)") for e in (0, 1, 2)]
    itens += [Line2D([], [], color='gray', lw=1.6, ls='--', label='EDO de campo médio (mesma cor)')]
    fig.legend(handles=itens, loc='lower center', ncol=4, fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.03))
    usados = [x[2] for x in paineis]
    fig.suptitle("Autômato estocástico e EDO de campo médio, sem vacina "
                 f"(média ± {FV.NOME_BANDA[banda]}, {FV.texto_seeds(resumos, usados)})", fontsize=14, y=1.0)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figura 15: semente inicial
# ---------------------------------------------------------------------------
def prob_extincao_ramificacao(par, heterogeneo=True):
    """Probabilidade de a linhagem de UM infectado se extinguir, em mistura homogênea.

    Processo de ramificação com as regras do modelo: um infectado transmite
    durante L passos, L geométrica com média 1/p_rec, e faz em cada passo um
    número de Poisson de infecções com média r = alpha (8 W_local + K W_mercado) <S0> T0.
    A função geradora do número de infecções é G(s) = p z / (1 - (1 - p) z),
    com z = exp(-r (1 - s)); a probabilidade de extinção q é a menor raiz de
    q = G(q). Com ``heterogeneo``, G é a média sobre T0 uniforme; sem ele, usa
    T0 = <T0>. Vale para população infinita e sem estrutura espacial: na grade,
    os vizinhos de um infectado se esgotam, e a extinção deve ser mais provável.
    """
    p = par["p_rec"]
    contatos = 8 * par["W_local"] + par["K_encontros"] * par["W_mercado"]
    a, b = par["atrib_min"], par["atrib_max"]
    T0 = np.linspace(a, b, 4001) if heterogeneo else np.array([0.5 * (a + b)])
    r = par["alpha"] * contatos * 0.5 * (a + b) * T0

    def g_menos_q(q):
        z = np.exp(-r * (1.0 - q))
        return float(np.mean(p * z / (1.0 - (1.0 - p) * z))) - q

    if g_menos_q(1.0 - 1e-9) >= 0:      # subcrítico: extinção certa
        return 1.0
    return float(brentq(g_menos_q, 0.0, 1.0 - 1e-9))


def configuracoes_semente(doenca):
    """Lista de (I0, condição, código) da varredura da semente, incluindo o caso principal I0 = 9."""
    b = f"{PREFIXO[doenca]}0"
    lista = [(N_INICIAIS, "C", f"{b}C"), (N_INICIAIS, "E", f"{b}E")]
    for lado in LADOS_SEMENTE:
        n = lado * lado
        lista.append((n, "C", f"{b}C_I{n}"))
        if n > 1:
            lista.append((n, "E", f"{b}E_I{n}"))
    return sorted(lista)


def _barra(valores, banda):
    """Média e meia largura (dp ou ep) de uma grandeza por seed."""
    e = FV.estat(valores)
    return float(e["media"]), float(np.nan_to_num(e[banda]))


def _painel_fracao_falhos(ax, pontos, resumos):
    """Fração de surtos falhos (com erro padrão binomial) para pontos (x, condição, código)."""
    for cond, marcador, desl in (("C", 'o', -0.12), ("E", 's', 0.12)):
        xs, ys, es, rot = [], [], [], []
        for x, c, codigo in pontos:
            if c != cond or codigo not in resumos:
                continue
            f = resumos[codigo]["falho_seeds"]
            xs.append(x + desl)
            ys.append(f.mean())
            es.append(np.sqrt(f.mean() * (1 - f.mean()) / f.size))
            rot.append(f"{int(f.sum())}/{f.size}")
        if xs:
            ax.errorbar(xs, ys, yerr=es, fmt=marcador, color='#333333', markersize=7, capsize=3,
                        mfc='#333333' if cond == "C" else 'white', linestyle='none')
            for x, y, e, txt in zip(xs, ys, es, rot):
                ax.annotate(txt, (x, y + e), textcoords='offset points', xytext=(0, 5), ha='center', fontsize=8.5)


def fig_semente(resumos, banda, t_max_flu=400):
    """Linha 1: Flu-like; linha 2: Measles-like. Colunas: I(t), surtos falhos, passo do pico."""
    dados = {d: [x for x in configuracoes_semente(d) if x[2] in resumos] for d in ("flu", "measles")}
    if not any(dados.values()):
        return None
    valores_i0 = sorted({x[0] for d in dados.values() for x in d})
    cor = {n: RAMPA[min(i, len(RAMPA) - 1)] for i, n in enumerate(valores_i0)}
    pos = {n: i for i, n in enumerate(valores_i0)}
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), gridspec_kw=dict(width_ratios=[1.7, 1, 1]))
    for i, doenca in enumerate(("flu", "measles")):
        lista = dados[doenca]
        ax = axes[i, 0]
        for n, cond, codigo in lista:
            r = resumos[codigo]
            ax.plot(np.arange(r["horizonte"] + 1), r["I"]["media"], '-' if cond == "C" else '--', color=cor[n],
                    linewidth=2)
        if lista:
            rs = [resumos[x[2]] for x in lista]
            ax.set_xlim(0, min(t_max_flu, rs[0]["horizonte"]) if doenca == "flu" else fim_da_epidemia(rs))
        ax.set_ylim(bottom=0)
        ax.set_title(f"{FV.NOME_DOENCA[doenca]} sem vacina: média de infectados", fontsize=12.5)
        ax.set_xlabel('Tempo (passos)', fontsize=12)
        ax.set_ylabel('Indivíduos infectados', fontsize=12)
        FV.estilo_eixo(ax)

        ax = axes[i, 1]
        _painel_fracao_falhos(ax, [(pos[n], c, cod) for n, c, cod in lista], resumos)
        if lista:
            par = dict(resumos[lista[0][2]]["param"])
            q_hom = prob_extincao_ramificacao(par, heterogeneo=False)
            q_het = prob_extincao_ramificacao(par, heterogeneo=True)
            for n in valores_i0:
                ax.plot([pos[n] - 0.3, pos[n] + 0.3], [q_het ** n] * 2, color='#D55E00', linewidth=2)
                ax.plot([pos[n] - 0.3, pos[n] + 0.3], [q_hom ** n] * 2, color='#D55E00', linewidth=1.5, linestyle=':')
            ax.set_title(f"Surtos falhos (menos de {100 * FV.LIMIAR_SURTO_FALHO:.0f}% da população infectada)\n"
                         f"ramificação: q = {q_het:.3f} (T₀ heterogêneo), {q_hom:.3f} (T₀ médio)", fontsize=11)
        ax.set_xticks(range(len(valores_i0)))
        ax.set_xticklabels([str(n) for n in valores_i0])
        ax.set_xlabel('Infectados iniciais I₀', fontsize=12)
        ax.set_ylabel('Fração das execuções', fontsize=12)
        ax.set_ylim(-0.02, max(0.45, ax.get_ylim()[1]))
        FV.estilo_eixo(ax)

        ax = axes[i, 2]
        for cond, marcador, desl in (("C", 'o', -0.12), ("E", 's', 0.12)):
            pts = [(pos[n] + desl, *_barra(resumos[cod]["pico_t_seeds"][~resumos[cod]["falho_seeds"]], banda))
                   for n, c, cod in lista if c == cond and (~resumos[cod]["falho_seeds"]).any()]
            if pts:
                xs, ys, es = zip(*pts)
                ax.errorbar(xs, ys, yerr=es, fmt=marcador, color='#333333', markersize=7, capsize=3,
                            mfc='#333333' if cond == "C" else 'white', linestyle='none')
        ax.set_xticks(range(len(valores_i0)))
        ax.set_xticklabels([str(n) for n in valores_i0])
        ax.set_xlabel('Infectados iniciais I₀', fontsize=12)
        ax.set_ylabel('Passo do pico máximo de infectados', fontsize=12)
        ax.set_title(f"Instante do pico, sem os surtos falhos (média ± {FV.NOME_BANDA[banda]})", fontsize=11)
        ax.set_ylim(bottom=0)
        FV.estilo_eixo(ax)

    itens = [Line2D([], [], color=cor[n], lw=2.5, label=f"I₀ = {n}") for n in valores_i0]
    itens += [Line2D([], [], color='#333333', lw=2, ls='-', marker='o', label='Bloco central'),
              Line2D([], [], color='#333333', lw=2, ls='--', marker='s', mfc='white', label='Espalhados'),
              Line2D([], [], color='#D55E00', lw=2, label='Ramificação: q^I₀, T₀ heterogêneo'),
              Line2D([], [], color='#D55E00', lw=1.5, ls=':', label='Ramificação: q^I₀, T₀ médio')]
    fig.legend(handles=itens, loc='lower center', ncol=len(itens), fontsize=10.5, frameon=False,
               bbox_to_anchor=(0.5, -0.035))
    usados = [x[2] for d in dados.values() for x in d]
    fig.suptitle(f"Efeito da semente inicial, sem vacina ({FV.texto_seeds(resumos, usados)})", fontsize=14, y=1.0)
    fig.text(0.5, -0.06, "Com I₀ = 1 só há a versão em bloco: um único infectado em uma grade periódica é "
             "estatisticamente o mesmo no centro ou em um sítio sorteado. Nos surtos falhos, o erro é o erro "
             "padrão binomial e o rótulo é a contagem.", ha='center', fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figura 16: conexões globais
# ---------------------------------------------------------------------------
def configuracoes_conexoes(doenca):
    """Lista de (K, condição, código), incluindo o caso principal K = 3."""
    b = f"{PREFIXO[doenca]}0"
    lista = []
    for cond in ("C", "E"):
        lista.append((3, cond, f"{b}{cond}"))
        lista += [(K, cond, f"{b}{cond}_K{K}") for K in K_VARREDURA]
    return sorted(lista)


def fig_conexoes(resumos, banda):
    """Linha 1: Flu-like; linha 2: Measles-like. Colunas: I(t), população já infectada, suscetíveis no fim."""
    dados = {d: [x for x in configuracoes_conexoes(d) if x[2] in resumos] for d in ("flu", "measles")}
    if not any(dados.values()):
        return None
    valores_k = sorted({x[0] for d in dados.values() for x in d})
    cor = {K: RAMPA[min(i, len(RAMPA) - 1)] for i, K in enumerate(valores_k)}
    pos = {K: i for i, K in enumerate(valores_k)}
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), gridspec_kw=dict(width_ratios=[1.7, 1, 1]))
    for i, doenca in enumerate(("flu", "measles")):
        lista = dados[doenca]
        ax = axes[i, 0]
        for K, cond, codigo in lista:
            r = resumos[codigo]
            t = np.arange(1, r["horizonte"] + 1)
            ax.plot(t, r["I"]["media"][1:], '-' if cond == "C" else '--', color=cor[K], linewidth=2)
        ax.set_xscale('log')
        ax.set_ylim(bottom=0)
        ax.set_title(f"{FV.NOME_DOENCA[doenca]} sem vacina: média de infectados (tempo em escala logarítmica)",
                     fontsize=12)
        ax.set_xlabel('Tempo (passos)', fontsize=12)
        ax.set_ylabel('Indivíduos infectados', fontsize=12)
        FV.estilo_eixo(ax)

        ax = axes[i, 1]
        for cond, marcador, desl in (("C", 'o', -0.12), ("E", 's', 0.12)):
            pts = [(pos[K] + desl, *_barra(100 * resumos[cod]["final_seeds"], banda))
                   for K, c, cod in lista if c == cond]
            if pts:
                xs, ys, es = zip(*pts)
                ax.errorbar(xs, ys, yerr=es, fmt=marcador, color='#333333', markersize=7, capsize=3,
                            mfc='#333333' if cond == "C" else 'white', linestyle='none')
        ax.set_ylim(0, 105)
        ax.set_ylabel('% da população', fontsize=12)
        ax.set_title(f"População infectada ao menos uma vez\n(média ± {FV.NOME_BANDA[banda]})", fontsize=11)

        ax = axes[i, 2]
        for cond, marcador, desl in (("C", 'o', -0.12), ("E", 's', 0.12)):
            for chave, cor_p in (("perc_p", '#56B4E9'), ("perc_pinf", '#0072B2')):
                pts = [(pos[K] + desl, *_barra(resumos[cod]["perc"][chave][:, -1], banda))
                       for K, c, cod in lista if c == cond and "perc" in resumos[cod]]
                if pts:
                    xs, ys, es = zip(*pts)
                    ax.errorbar(xs, ys, yerr=es, fmt=marcador, color=cor_p, markersize=7, capsize=3,
                                mfc=cor_p if cond == "C" else 'white', linestyle='none')
        ax.axhline(P_C_MOORE, color='gray', linestyle='--', linewidth=1.2)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel('Fração dos sítios', fontsize=12)
        horizonte = resumos[lista[0][2]]["horizonte"] if lista else 0
        ax.set_title(f"Suscetíveis em t = {horizonte}: fração p e fração no maior aglomerado P∞\n"
                     f"(média ± {FV.NOME_BANDA[banda]})", fontsize=11)
        for ax in axes[i, 1:]:
            ax.set_xticks(range(len(valores_k)))
            ax.set_xticklabels([str(K) for K in valores_k])
            ax.set_xlabel('Encontros de mercado por passo, K', fontsize=12)
            FV.estilo_eixo(ax)

    itens = [Line2D([], [], color=cor[K], lw=2.5, label=f"K = {K}") for K in valores_k]
    itens += [Line2D([], [], color='#333333', lw=2, ls='-', marker='o', label='Bloco central'),
              Line2D([], [], color='#333333', lw=2, ls='--', marker='s', mfc='white', label='Espalhados'),
              Line2D([], [], color='#56B4E9', lw=0, marker='o', label='p (suscetíveis)'),
              Line2D([], [], color='#0072B2', lw=0, marker='o', label='P∞ (maior aglomerado)'),
              Line2D([], [], color='gray', lw=1.2, ls='--', label=f"p_c = {P_C_MOORE:.3f}".replace('.', ','))]
    fig.legend(handles=itens, loc='lower center', ncol=len(itens), fontsize=10.5, frameon=False,
               bbox_to_anchor=(0.5, -0.035))
    usados = [x[2] for d in dados.values() for x in d]
    fig.suptitle(f"Influência das conexões globais, sem vacina ({FV.texto_seeds(resumos, usados)})",
                 fontsize=14, y=1.0)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figura 17: regimes de vacinação
# ---------------------------------------------------------------------------
def fig_regimes(resumos, cond, banda=None, t_zoom=400):
    """Preventiva, reativa e contínua em um só painel por doença.

    Sem ``banda``: as sete curvas (alocação aleatória em linha cheia, gaussiana
    tracejada), sem sombreado. Com ``banda`` ("dp" ou "ep"): só as curvas de
    alocação aleatória e a sem vacina, com a banda sombreada. O controle sem
    reforço (F7) não entra: ele responde a outra pergunta e está na figura 05.
    """
    fam = FV.por_condicao(resumos, cond)
    if not fam:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    paineis = (("flu", f"Flu-like: primeiros {t_zoom} passos", "zoom"),
               ("flu", "Flu-like: simulação inteira", "tudo"),
               ("measles", "Measles-like", "fim"))
    usados = []
    for ax, (doenca, titulo, janela) in zip(axes, paineis):
        desenhados = []
        for base in bases(doenca, fam):
            c = CENARIOS[base]
            if c.get("reforco") is False:
                continue
            gaussiana = c.get("alocacao") == "gaussiana"
            if banda and gaussiana:
                continue
            r = fam[base]
            t = np.arange(r["horizonte"] + 1)
            cor = COR_REGIME[regime_do_cenario(base)]
            if banda:
                FV.curva_com_banda(ax, t, r["I"], banda, cor, None)
            else:
                ax.plot(t, r["I"]["media"], '--' if gaussiana else '-', color=cor, linewidth=2)
            desenhados.append(r)
            usados.append(base)
        ax.axvline(60, color='gray', linestyle='--', linewidth=1.2)
        if desenhados:
            horizonte = desenhados[0]["horizonte"]
            limite = {"zoom": min(t_zoom, horizonte), "tudo": horizonte, "fim": fim_da_epidemia(desenhados)}[janela]
            ax.set_xlim(0, limite)
            if janela == "fim" and limite < horizonte:
                titulo += f" (eixo cortado em t = {limite})"
        ax.set_ylim(bottom=0)
        ax.set_title(titulo, fontsize=12.5)
        ax.set_xlabel('Tempo (passos)', fontsize=12)
        FV.estilo_eixo(ax)
    if not usados:
        plt.close(fig)
        return None
    axes[0].set_ylabel('Indivíduos infectados (média entre seeds)', fontsize=12)
    itens = [Line2D([], [], color=COR_REGIME[k], lw=2.5, label=NOME_REGIME[k])
             for k in ("nenhum", "t0", "t60", "continuo")]
    if not banda:
        itens += [Line2D([], [], color='#333333', lw=2, ls='-', label='Alocação aleatória'),
                  Line2D([], [], color='#333333', lw=2, ls='--', label='Alocação gaussiana (grupos de risco)')]
    itens.append(Line2D([], [], color='gray', lw=1.2, ls='--', label='t = 60 (vacinação reativa)'))
    fig.legend(handles=itens, loc='lower center', ncol=len(itens), fontsize=10.5, frameon=False,
               bbox_to_anchor=(0.5, -0.07))
    como = f"média ± {FV.NOME_BANDA[banda]}, só alocação aleatória" if banda else "médias, sem banda"
    fig.suptitle(f"Vacinação preventiva, reativa e contínua ({como}; {FV.texto_seeds(fam, usados)}; "
                 f"infectados iniciais: {FV.NOME_CONDICAO[cond]})", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figura 18: percolação dos suscetíveis
# ---------------------------------------------------------------------------
_REFERENCIA = {}


def referencia_diluicao_aleatoria(N, n_p=40, repeticoes=12):
    """P∞(p) e probabilidade de envolver o toro quando os sítios são ocupados ao acaso.

    Monte Carlo na mesma rede (N x N, periódica, vizinhança de Moore): é a curva
    de percolação de sítios sem correlações, com a qual comparar os suscetíveis
    que sobram de uma epidemia. A seed é fixa para a figura ser reprodutível.
    """
    if N not in _REFERENCIA:
        rng = np.random.default_rng(12345)
        ps = np.linspace(0.02, 1.0, n_p)
        pinf, env = np.zeros(n_p), np.zeros(n_p)
        for i, p in enumerate(ps):
            med = [medidas_percolacao(rng.random((N, N)) < p) for _ in range(repeticoes)]
            pinf[i] = np.mean([m["p_inf"] for m in med])
            env[i] = np.mean([m["envolve"] for m in med])
        _REFERENCIA[N] = (ps, pinf, env)
    return _REFERENCIA[N]


def _grupo_alocacao(base):
    """(rótulo, cor) do tipo de cenário, com as cores das figuras 05 a 12."""
    c = CENARIOS[base]
    if c.get("regime") == "nenhum":
        return "Sem vacina", FV.COR_SEM_VACINA
    if c.get("alocacao") == "gaussiana":
        return "Alocação gaussiana", FV.COR_GAUSSIANA
    return "Alocação aleatória", FV.COR_ALEATORIA


def fig_percolacao(resumos, cond, largura_classe=0.04):
    """Parâmetro de ordem P∞ contra a fração de suscetíveis p, e probabilidade de envolver o toro.

    Cada ponto é uma seed em um instante gravado de um cenário. A curva cinza é
    a percolação de sítios ocupados ao acaso na mesma rede.
    """
    fam = FV.por_condicao(resumos, cond)
    fam = {b: r for b, r in fam.items() if "perc" in r}
    if not fam:
        return None
    N = next(iter(fam.values()))["param"]["N"]
    ps, pinf_ref, env_ref = referencia_diluicao_aleatoria(N)
    fig, axes = plt.subplots(2, 2, figsize=(17, 10), sharex=True, sharey='row')
    bordas = np.arange(0.0, 1.0 + largura_classe, largura_classe)
    centros = 0.5 * (bordas[1:] + bordas[:-1])
    rotulos = {}
    for j, doenca in enumerate(("flu", "measles")):
        grupos = {}
        for base in bases(doenca, fam):
            nome, cor = _grupo_alocacao(base)
            rotulos[nome] = cor
            g = grupos.setdefault(nome, dict(cor=cor, p=[], pinf=[], env=[]))
            pc = fam[base]["perc"]
            g["p"].append(pc["perc_p"].ravel())
            g["pinf"].append(pc["perc_pinf"].ravel())
            g["env"].append(pc["perc_envolve"].ravel())
        ax, ax2 = axes[0, j], axes[1, j]
        for nome, g in grupos.items():
            p, pinf, env = (np.concatenate(g[k]) for k in ("p", "pinf", "env"))
            # muitos pontos: mais transparência, para a densidade aparecer
            ax.scatter(p, pinf, s=7, color=g["cor"], alpha=float(np.clip(1500 / p.size, 0.06, 0.6)), linewidths=0,
                       rasterized=True)
            classe = np.digitize(p, bordas) - 1
            frac = np.array([env[classe == k].mean() if (classe == k).sum() >= 5 else np.nan
                             for k in range(centros.size)])
            # classes sem medidas ficam em branco: a linha não liga pontos separados por um vazio
            ax2.plot(centros, frac, 'o-', color=g["cor"], markersize=4, linewidth=1.5)
        ax.plot(ps, pinf_ref, color='#555555', linewidth=2)
        ax.plot([0, 1], [0, 1], color='#999999', linewidth=1, linestyle=':')
        ax2.plot(ps, env_ref, color='#555555', linewidth=2)
        for a in (ax, ax2):
            a.axvline(P_C_MOORE, color='gray', linestyle='--', linewidth=1.2)
            a.set_xlim(0, 1)
            FV.estilo_eixo(a)
        ax.set_ylim(0, 1)
        ax2.set_ylim(-0.03, 1.03)
        ax.set_title(f"{FV.NOME_DOENCA[doenca]}: parâmetro de ordem", fontsize=13)
        ax2.set_title(f"{FV.NOME_DOENCA[doenca]}: fração das medidas em que o maior aglomerado envolve o toro "
                      f"(classes de p de largura {largura_classe:.2f})".replace('.', ','), fontsize=11.5)
        ax2.set_xlabel('Fração de sítios suscetíveis, p', fontsize=12)
    axes[0, 0].set_ylabel('Fração dos sítios no maior aglomerado, P∞', fontsize=12)
    axes[1, 0].set_ylabel('Fração das medidas', fontsize=12)
    itens = [Line2D([], [], color=cor, lw=0, marker='o', label=nome) for nome, cor in rotulos.items()]
    itens += [Line2D([], [], color='#555555', lw=2, label='Sítios ocupados ao acaso (mesma rede)'),
              Line2D([], [], color='gray', lw=1.2, ls='--', label=f"p_c = {P_C_MOORE:.3f} (rede infinita)"
                     .replace('.', ',')),
              Line2D([], [], color='#999999', lw=1, ls=':', label='P∞ = p (todos os suscetíveis ligados)')]
    fig.legend(handles=itens, loc='lower center', ncol=len(itens), fontsize=10.5, frameon=False,
               bbox_to_anchor=(0.5, -0.035))
    tempos = ", ".join(str(int(t)) for t in next(iter(fam.values()))["perc"]["perc_tempos"])
    fig.suptitle(f"Percolação dos suscetíveis: cada ponto é uma seed em um dos instantes t = {tempos} "
                 f"({FV.texto_seeds(fam, list(fam))}; infectados iniciais: {FV.NOME_CONDICAO[cond]})",
                 fontsize=13.5, y=1.0)
    fig.tight_layout()
    return fig


def fig_percolacao_mapas(resumos, cond, cenarios=("M1", "M2", "F1", "F2"), tempos=(10, 150, None)):
    """Estados em três instantes e, na última coluna, os aglomerados de suscetíveis do último instante."""
    fam = FV.por_condicao(resumos, cond)
    usados = [b for b in cenarios if b in fam and "grades_estado" in fam[b]["grades"]
              and fam[b]["grades"]["grades_seeds"].size]
    if not usados:
        return None
    horizonte = fam[usados[0]]["horizonte"]
    guardados = [int(t) for t in fam[usados[0]]["grades"]["grades_tempos"]]
    tempos = [horizonte if t is None else int(t) for t in tempos]
    tempos = [t for t in dict.fromkeys(tempos) if t in guardados]
    cmap = ListedColormap(CORES_ESTADOS)
    cmap_aglom = ListedColormap([COR_INERTE, '#56B4E9', '#0072B2', CORES_ESTADOS[INFECTADO]])
    ncol = len(tempos) + 1
    fig, axes = plt.subplots(len(usados), ncol, figsize=(3.6 * ncol + 0.5, 3.75 * len(usados) + 1.0), squeeze=False)
    for i, base in enumerate(usados):
        r = fam[base]
        k, seed = seed_representativa(r)
        for j, t in enumerate(tempos):
            ax = axes[i, j]
            ax.imshow(r["grades"]["grades_estado"][k, guardados.index(t)], cmap=cmap, vmin=-0.5, vmax=3.5,
                      interpolation='nearest')
            ax.set_title(f"Estados em t = {t}", fontsize=11.5)
            if j == 0:
                ax.set_ylabel(f"{descricao(base)}\n(seed {seed})", fontsize=10.5)
        estado = r["grades"]["grades_estado"][k, guardados.index(tempos[-1])]
        susc = estado == SUSCETIVEL
        rot, tam, env = aglomerados_periodicos(susc)
        mapa = np.zeros(estado.shape, dtype=int)
        mapa[susc] = 1
        texto = "sem suscetíveis"
        if tam.size:
            maior = int(np.argmax(tam))
            mapa[rot == maior + 1] = 2
            texto = (f"p = {susc.mean():.2f}, P∞ = {tam[maior] / susc.size:.2f}, "
                     f"{'envolve' if env[maior] else 'não envolve'} o toro").replace('.', ',')
        mapa[estado == INFECTADO] = 3
        ax = axes[i, -1]
        ax.imshow(mapa, cmap=cmap_aglom, vmin=-0.5, vmax=3.5, interpolation='nearest')
        ax.set_title(f"Aglomerados em t = {tempos[-1]}\n{texto}", fontsize=10.5)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    itens = [Patch(facecolor=CORES_ESTADOS[e], edgecolor='none', label=NOMES_ESTADOS[e]) for e in range(4)]
    itens += [Patch(facecolor='#0072B2', edgecolor='none', label='Maior aglomerado de suscetíveis'),
              Patch(facecolor='#56B4E9', edgecolor='none', label='Outros suscetíveis'),
              Patch(facecolor=COR_INERTE, edgecolor='none', label='Material inerte (recuperados e vacinados)')]
    fig.suptitle("Aglomerados de suscetíveis depois da vacinação em t = 0 "
                 f"(infectados iniciais: {FV.NOME_CONDICAO[cond]})", fontsize=14, y=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 0.985), h_pad=1.6)
    fig.legend(handles=itens, loc='lower center', ncol=4, fontsize=10.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    return fig


# ---------------------------------------------------------------------------
# Figura 19: bloco central (C) contra espalhados (E)
# ---------------------------------------------------------------------------
def diferenca_pareada(rc, re):
    """Estatísticas de I_E(t) - I_C(t), seed a seed (mesma população nas duas condições)."""
    _, ic, ie = np.intersect1d(rc["seeds"], re["seeds"], return_indices=True)
    return FV.estat(re["I_seeds"][ie].astype(np.float32) - rc["I_seeds"][ic])


def fig_diferenca_condicoes(resumos, doenca, banda, t_max_flu=600):
    """Um painel por cenário com a diferença pareada I_E(t) - I_C(t) e sua banda."""
    pares = [(b, resumos[f"{b}C"], resumos[f"{b}E"]) for b in CENARIOS
             if b.startswith(PREFIXO[doenca]) and f"{b}C" in resumos and f"{b}E" in resumos]
    if not pares:
        return None
    ncol = 4
    nlin = int(np.ceil(len(pares) / ncol))
    fig, axes = plt.subplots(nlin, ncol, figsize=(18, 4.4 * nlin), squeeze=False, sharex=True)
    for ax in axes.ravel()[len(pares):]:
        ax.set_visible(False)
    todos = [r for _, rc, re in pares for r in (rc, re)]
    horizonte = todos[0]["horizonte"]
    limite = min(t_max_flu, horizonte) if doenca == "flu" else fim_da_epidemia(todos)
    for ax, (base, rc, re) in zip(axes.ravel(), pares):
        d = diferenca_pareada(rc, re)
        t = np.arange(d["media"].size)
        FV.curva_com_banda(ax, t, d, banda, '#333333', None)
        ax.axhline(0, color='gray', linewidth=1)
        ax.set_xlim(0, limite)
        ax.set_title(descricao(base), fontsize=12)
        FV.estilo_eixo(ax)
    for k, ax in enumerate(axes.ravel()[:len(pares)]):
        if k + ncol >= len(pares):        # último painel visível da coluna: mostra o eixo do tempo
            ax.set_xlabel('Tempo (passos)', fontsize=12)
            ax.tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel('I espalhados − I bloco central', fontsize=11.5)
    corte = f"primeiros {limite} passos" if limite < horizonte else "simulação inteira"
    fig.suptitle(f"{FV.NOME_DOENCA[doenca]}: diferença de infectados entre 9 espalhados e 9 em bloco central, "
                 f"seed a seed (média ± {FV.NOME_BANDA[banda]}, {FV.texto_seeds(resumos, [b + 'C' for b, _, _ in pares])}; "
                 f"{corte})", fontsize=13.5, y=1.0)
    fig.text(0.5, -0.02, "Acima de zero: mais infectados com a semente espalhada naquele instante. As duas condições "
             "usam a mesma população (S₀ e T₀) em cada seed.", ha='center', fontsize=10)
    fig.tight_layout()
    return fig


def fig_metricas_condicoes(resumos, banda):
    """Tamanho final, pico, instante do pico e surtos falhos, lado a lado em C e E, para todos os cenários."""
    linhas = {d: [b for b in CENARIOS if b.startswith(PREFIXO[d]) and f"{b}C" in resumos and f"{b}E" in resumos]
              for d in ("flu", "measles")}
    if not any(linhas.values()):
        return None
    metricas = (("final_seeds", 100.0, "% da população infectada ao menos uma vez"),
                ("pico_altura_seeds", 1.0, "Pico máximo de infectados"),
                ("pico_t_seeds", 1.0, "Passo do pico máximo"),
                ("falho_seeds", 1.0, "Fração de surtos falhos"))
    fig, axes = plt.subplots(2, 4, figsize=(19, 9), squeeze=False)
    for i, doenca in enumerate(("flu", "measles")):
        lista = linhas[doenca]
        x = np.arange(len(lista))
        for j, (chave, fator, nome) in enumerate(metricas):
            ax = axes[i, j]
            for cond, marcador, desl in (("C", 'o', -0.15), ("E", 's', 0.15)):
                ys, es = [], []
                for b in lista:
                    v = resumos[f"{b}{cond}"][chave].astype(float) * fator
                    if chave == "falho_seeds":      # proporção: erro padrão binomial nas duas versões
                        ys.append(v.mean())
                        es.append(np.sqrt(v.mean() * (1 - v.mean()) / v.size))
                    else:
                        m, e = _barra(v, banda)
                        ys.append(m)
                        es.append(e)
                if lista:
                    ax.errorbar(x + desl, ys, yerr=es, fmt=marcador, color='#333333', markersize=6, capsize=3,
                                mfc='#333333' if cond == "C" else 'white', linestyle='none')
            ax.set_xticks(x)
            ax.set_xticklabels(lista, fontsize=10.5)
            ax.set_title(f"{FV.NOME_DOENCA[doenca]}: {nome[0].lower()}{nome[1:]}", fontsize=11.5)
            ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
            if chave == "falho_seeds":
                ax.set_ylim(-0.03, 1.03)
            FV.estilo_eixo(ax)
        for ax in axes[i]:
            ax.set_xlabel('Cenário', fontsize=11.5)
    itens = [Line2D([], [], color='#333333', lw=0, marker='o', label='9 em bloco central (C)'),
             Line2D([], [], color='#333333', lw=0, marker='s', mfc='white', label='9 espalhados (E)')]
    fig.legend(handles=itens, loc='lower center', ncol=2, fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.04))
    usados = [f"{b}C" for d in linhas.values() for b in d]
    fig.suptitle("Bloco central contra espalhados: métricas por cenário "
                 f"(média ± {FV.NOME_BANDA[banda]}; {FV.texto_seeds(resumos, usados)})", fontsize=14, y=1.0)
    fig.text(0.5, -0.065, f"Surto falho: menos de {100 * FV.LIMIAR_SURTO_FALHO:.0f}% da população infectada; o erro "
             "dessa fração é o erro padrão binomial nas duas versões da figura.", ha='center', fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
def todas(resumos, gravar, escolhidas=None):
    """Grava as figuras 13 a 19 com ``gravar(fig, nome)``; ``escolhidas`` filtra pelos números."""
    def quer(numero):
        return not escolhidas or numero in escolhidas

    if quer("13"):
        for doenca in ("flu", "measles"):
            for cond in ("C", "E"):
                gravar(fig_evolucao_espacial(resumos, doenca, cond), f"13_evolucao_espacial_{doenca}_{cond}.png")
    for banda in BANDAS:
        if quer("14"):
            gravar(fig_edo(resumos, banda), f"14_comparacao_edo_{banda}.png")
        if quer("15"):
            gravar(fig_semente(resumos, banda), f"15_semente_inicial_{banda}.png")
        if quer("16"):
            gravar(fig_conexoes(resumos, banda), f"16_conexoes_globais_{banda}.png")
    for cond in ("C", "E"):
        if quer("17"):
            gravar(fig_regimes(resumos, cond), f"17_regimes_vacinacao_{cond}.png")
            for banda in BANDAS:
                gravar(fig_regimes(resumos, cond, banda), f"17_regimes_vacinacao_bandas_{banda}_{cond}.png")
        if quer("18"):
            gravar(fig_percolacao(resumos, cond), f"18_percolacao_parametro_de_ordem_{cond}.png")
            gravar(fig_percolacao_mapas(resumos, cond), f"18_percolacao_mapas_{cond}.png")
    if quer("19"):
        for banda in BANDAS:
            for doenca in ("flu", "measles"):
                gravar(fig_diferenca_condicoes(resumos, doenca, banda),
                       f"19_central_vs_espalhado_{doenca}_{banda}.png")
            gravar(fig_metricas_condicoes(resumos, banda), f"19_central_vs_espalhado_metricas_{banda}.png")
