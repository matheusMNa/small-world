"""
Testes do modelo de vacinação (seção 8 da especificação) e das extensões
posteriores: condição inicial espalhada, estados, percolação e configurações.

Python puro, sem dependências além das do projeto. A partir da raiz::

    python tests/test_model_vacinacao.py            # roda todos e imprime o resultado de cada um
    python tests/test_model_vacinacao.py 4 7        # só os testes 4 e 7

As funções também são reconhecidas pelo pytest, se ele estiver instalado.
"""
import os
import sys
import time
import traceback

import numpy as np
from scipy.optimize import brentq
from scipy.signal import convolve2d

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from analyses.ensemble import rodar_pedaco
from src.model import simulate
from src.model_vacinacao import (CENARIOS, CONFIGURACOES, DESLOCAMENTOS, INFECTADO, RECUPERADO, SUSCETIVEL,
                                 VACINADO, Simulacao, aglomerados_periodicos, codigos_do_grupo,
                                 contribuicoes_locais, curva_dose, indices_vizinhos, instante_pico,
                                 mascara_passos, medidas_percolacao, parametros_cenario, simular,
                                 taxas_por_faixa)


def test_01_contribuicoes_locais():
    """A soma das 8 contribuições feitas com np.roll é igual à convolução do modelo original."""
    rng = np.random.default_rng(123)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    for N in (7, 100):
        T = rng.random((N, N))
        pilha = contribuicoes_locais(T)
        assert pilha.shape == (8, N, N)
        assert np.allclose(pilha.sum(axis=0), convolve2d(T, kernel, mode='same', boundary='wrap'))
        # a pilha e a tabela de índices dos vizinhos descrevem os mesmos sítios
        assert np.array_equal(pilha.reshape(8, -1), T.ravel()[indices_vizinhos(N)])
    assert len(set(DESLOCAMENTOS)) == 8 and (0, 0) not in DESLOCAMENTOS


def test_02_equivalencia_com_modelo_atual():
    """M0 tem a mesma lei estatística que o simulate atual (50 seeds, 150 passos)."""
    n, passos = 50, 150
    I_velho, tot_velho, I_novo, tot_novo = [], [], [], []
    for s in range(n):
        _, I_hist, R_hist, _ = simulate(steps=passos, seed=s)
        I_velho.append(I_hist[passos])
        tot_velho.append(I_hist[passos] + R_hist[passos])   # sem vacina: já infectados
        r = simular(parametros_cenario("M0", horizonte=passos), s)
        I_novo.append(r["n_inf_faixa"][passos].sum())
        tot_novo.append(r["novas_inf_faixa"].sum() + 9)     # 9 infectados iniciais
    for nome, a, b in (("I(150)", I_velho, I_novo), ("total de infectados", tot_velho, tot_novo)):
        a, b = np.array(a, float), np.array(b, float)
        dif = abs(a.mean() - b.mean())
        ep = np.sqrt(a.var(ddof=1) / n + b.var(ddof=1) / n)
        print(f"      {nome}: atual {a.mean():.1f}, novo {b.mean():.1f}, "
              f"diferença {dif:.1f}, 2 erros padrão = {2 * ep:.1f}")
        assert dif <= 2 * ep, f"{nome}: diferença {dif:.1f} maior que 2 erros padrão ({2 * ep:.1f})"


def test_03_conservacao():
    """Toda infecção tem exatamente um transmissor, com contribuição positiva."""
    for codigo in ("F0", "F1", "F4", "F6", "M1", "M5"):
        for seed in (0, 1):
            sim = Simulacao(parametros_cenario(codigo, horizonte=250), seed)
            r = sim.rodar()
            # passo a passo e no total
            assert np.array_equal(r["novas_inf_faixa"].sum(axis=1), r["trans_faixa"].sum(axis=1))
            total = int(r["novas_inf_faixa"].sum())
            assert total > 0, "o teste precisa de infecções para ter sentido"
            assert int(sim.n_infec.sum()) - sim.n_iniciais == total == int(r["trans_faixa"].sum())
            assert np.all(sim._ultima_contrib_fonte > 0)
            # o histograma soma a população inteira
            assert int(r["hist_infeccoes"].sum()) == sim.p.N ** 2


def test_04_curva_de_bateman():
    """O pico vale 1 em tau* e as quedas batem com os valores de referência."""
    referencia = {100.0: (10.8, (24, 43, 83)), 150.0: (12.0, (31, 58, 119))}
    for tau_d, (pico_ref, quedas_ref) in referencia.items():
        ts = instante_pico(3.0, tau_d)
        assert abs(ts - pico_ref) < 0.05, (tau_d, ts)
        assert abs(float(curva_dose(ts, 3.0, tau_d)) - 1.0) < 1e-12
        grade = np.linspace(0, 600, 60001)
        assert curva_dose(grade, 3.0, tau_d).max() <= 1.0 + 1e-12
        for nivel, ref in zip((0.9, 0.75, 0.5), quedas_ref):
            t = brentq(lambda x: float(curva_dose(x, 3.0, tau_d)) - nivel, ts, 2000)
            assert abs(t - ref) <= 0.5, (tau_d, nivel, t, ref)
    # limite sem decaimento (Measles-like): 1 - exp(-tau/tau_r)
    tau = np.array([0.0, 3.0, 30.0])
    assert np.allclose(curva_dose(tau, 3.0, np.inf), 1 - np.exp(-tau / 3.0))
    assert float(curva_dose(-1.0, 3.0, 100.0)) == 0.0


def test_05_regra_de_reforco():
    """Reforço só depois do pico, com m < 0,5 e em não infectados; nenhum em Measles-like nem em F7."""
    for codigo in ("F1", "F6"):
        sim = Simulacao(parametros_cenario(codigo, horizonte=400), 0)
        r = sim.rodar(auditoria=True)
        assert len(sim.auditoria_reforcos) > 0, "o teste precisa de reforços para ter sentido"
        n = 0
        for ev in sim.auditoria_reforcos:
            assert np.all(ev["tau"] > ev["tpico"]), "reforço durante a subida"
            assert np.all(ev["m"] < 0.5), "reforço com m >= 0,5"
            assert not ev["infectado"].any(), "reforço em infectado"
            n += ev["tau"].size
        assert n == int(r["reforcos"].sum())
        # os reforços não entram na conta de primeiras doses
        assert int(r["doses1"].sum()) == int((sim.n_doses > 0).sum())
        assert int(sim.n_doses.sum()) == int(r["doses1"].sum()) + n
    for codigo in ("F7", "M1", "M3", "M5", "F0"):
        r = simular(parametros_cenario(codigo, horizonte=400), 0)
        assert int(r["reforcos"].sum()) == 0, codigo


def test_06_doses_iguais():
    """Aleatória e gaussiana aplicam o mesmo número de primeiras doses (mesma seed, mesmo regime)."""
    pares = (("F1", "F2"), ("F3", "F4"), ("F5", "F6"), ("M1", "M2"), ("M3", "M4"), ("M5", "M6"))
    for seed in (0, 1, 2):
        doses = {}
        for a, b in pares:
            ra = simular(parametros_cenario(a, horizonte=130), seed)
            rb = simular(parametros_cenario(b, horizonte=130), seed)
            assert np.array_equal(ra["doses1"], rb["doses1"]), (a, b, seed)
            doses[a] = int(ra["doses1"].sum())
            assert doses[a] > 0
            if a in ("F3", "M3"):
                # até a vacinação em t = 60 a dinâmica é idêntica nos dois cenários
                for k in ("n_inf_faixa", "novas_inf_faixa", "trans_faixa"):
                    assert np.array_equal(ra[k][:60], rb[k][:60]), (a, b, seed, k)
        # contínuo: exatamente 5000, 50 por passo; t = 60 aplica menos que t = 0
        assert doses["F5"] == 5000 and doses["M5"] == 5000
        assert doses["F3"] < doses["F1"] and doses["M3"] < doses["M1"]
        r = simular(parametros_cenario("M5", horizonte=130), seed)
        assert np.all(r["doses1"][:100] == 50) and np.all(r["doses1"][100:] == 0)


def test_07_alocacao_gaussiana():
    """Com 50% de cobertura: média de S0 dos vacinados perto de 0,78 e ~88% das doses em S0 >= 0,6."""
    medias, fracoes, medias_aleat = [], [], []
    for seed in range(5):
        sim = Simulacao(parametros_cenario("M2", horizonte=1), seed)
        sim.rodar()
        vac = sim.n_doses > 0
        medias.append(sim.S0[vac].mean())
        fracoes.append((sim.S0[vac] >= 0.6).mean())
        sim = Simulacao(parametros_cenario("M1", horizonte=1), seed)
        sim.rodar()
        medias_aleat.append(sim.S0[sim.n_doses > 0].mean())
    print(f"      gaussiana: <S0> = {np.mean(medias):.3f}, doses em S0 >= 0,6 = {np.mean(fracoes):.1%}; "
          f"aleatória: <S0> = {np.mean(medias_aleat):.3f}")
    assert abs(np.mean(medias) - 0.78) < 0.01
    assert abs(np.mean(fracoes) - 0.88) < 0.02
    assert abs(np.mean(medias_aleat) - 0.60) < 0.01


def test_08_faixa_protegida():
    """Em M1, a taxa de infecção da faixa m >= 0,90 é no máximo ~10% da dos nunca vacinados."""
    # soma numeradores e denominadores de 4 seeds antes de dividir, com a mesma
    # máscara de passos usada nas métricas (só passos com infectados)
    infeccoes = np.zeros(5)
    pessoas_passo = np.zeros(5)
    for seed in range(4):
        r = simular(parametros_cenario("M1", horizonte=600), seed)
        mask = mascara_passos(r["n_inf_faixa"])[:, None]
        infeccoes += (r["novas_inf_faixa"] * mask).sum(axis=0)
        pessoas_passo += (r["n_naoinf_faixa"] * mask).sum(axis=0)
        # a função de métricas dá o mesmo resultado para uma execução
        lam_1, _ = taxas_por_faixa(r)
        esperado = (r["novas_inf_faixa"] * mask).sum(axis=0) / np.maximum((r["n_naoinf_faixa"] * mask).sum(axis=0), 1)
        ok = (r["n_naoinf_faixa"] * mask).sum(axis=0) > 0
        assert np.allclose(lam_1[ok], esperado[ok]) and np.all(np.isnan(lam_1[~ok]))
    lam_inf = infeccoes / pessoas_passo
    print(f"      taxa nunca vacinados = {lam_inf[0]:.2e}, taxa m >= 0,90 = {lam_inf[1]:.2e}, "
          f"razão = {lam_inf[1] / lam_inf[0]:.3f}")
    assert lam_inf[0] > 0
    assert lam_inf[1] <= 0.10 * lam_inf[0]


def test_09_measles_sem_reinfeccao():
    """No Measles-like nenhum indivíduo se infecta mais de uma vez."""
    for codigo in ("M0", "M1", "M2", "M3", "M5", "M6"):
        sim = Simulacao(parametros_cenario(codigo, horizonte=400), 0)
        r = sim.rodar()
        assert sim.n_infec.max() <= 1, codigo
        assert r["hist_infeccoes"][2:].sum() == 0
    # controle: no Flu-like há reinfecção, então o teste acima não é vazio
    sim = Simulacao(parametros_cenario("F0", horizonte=400), 0)
    sim.rodar()
    assert sim.n_infec.max() >= 2


def test_10_reprodutibilidade():
    """Mesma seed e mesmo cenário dão resultados idênticos, juntos ou separados no pedaço."""
    for codigo in ("F2", "M5"):
        a = simular(parametros_cenario(codigo, horizonte=200), 3, snapshots=(0, 60, 200))
        b = simular(parametros_cenario(codigo, horizonte=200), 3, snapshots=(0, 60, 200))
        for k in a:
            assert np.array_equal(a[k], b[k]), (codigo, k)
        # seeds diferentes dão realizações diferentes
        c = simular(parametros_cenario(codigo, horizonte=200), 4)
        assert not np.array_equal(a["n_inf_faixa"], c["n_inf_faixa"])

    # independência do pedaço: seeds 0 e 1 juntas e separadas
    for codigo in ("F1", "M3"):
        juntas = rodar_pedaco(codigo, [0, 1], 200)
        so0 = rodar_pedaco(codigo, [0], 200)
        so1 = rodar_pedaco(codigo, [1], 200)
        for k in juntas:
            if k in ("tempo_s", "seeds", "codigo", "horizonte", "parametros_json",
                     "grades_seeds", "grades_tempos", "perc_tempos"):
                continue
            assert np.array_equal(juntas[k][0], so0[k][0]), (codigo, k)
            assert np.array_equal(juntas[k][1], so1[k][0]), (codigo, k)

    # a seed k produz os mesmos S0 e T0 em todos os cenários
    s_f = Simulacao(parametros_cenario("F2"), 7)
    s_m = Simulacao(parametros_cenario("M5"), 7)
    assert np.array_equal(s_f.S0, s_m.S0) and np.array_equal(s_f.T0, s_m.T0)


# Resumos de execuções feitas com a versão do modelo anterior às extensões
# (três fluxos aleatórios, só bloco central): (infecções, I(100), I(horizonte),
# primeiras doses, reforços, transmissões por vacinados, nunca infectados, passo de extinção).
REFERENCIA_BLOCO_CENTRAL = {
    ("F1", 3, 250): (5581, 74, 967, 4996, 8838, 1041, 5275, -1),
    ("F6", 1, 250): (8753, 1065, 515, 5000, 5501, 1833, 3617, -1),
    ("M2", 5, 250): (90, 5, 0, 4996, 0, 1, 9901, 244),
    ("F4", 2, 200): (10475, 1339, 541, 3853, 2895, 1033, 2513, -1),
    ("M5", 7, 300): (4572, 863, 0, 5000, 0, 11, 5419, 285),
    ("F0", 0, 200): (16479, 2565, 946, 0, 0, 0, 228, -1),
}


def _resumo(r, h):
    return (int(r["novas_inf_faixa"].sum()), int(r["n_inf_faixa"][100].sum()), int(r["n_inf_faixa"][h].sum()),
            int(r["doses1"].sum()), int(r["reforcos"].sum()), int(r["trans_faixa"][:, 1:].sum()),
            int(r["hist_infeccoes"][0]), int(r["t_extincao"]))


def test_11_regressao_bloco_central():
    """Com bloco central, os resultados são idênticos aos da versão anterior do modelo."""
    for (codigo, seed, h), esperado in REFERENCIA_BLOCO_CENTRAL.items():
        for cod in (codigo, codigo + "C"):       # "F1" e "F1C" são a mesma configuração
            r = simular(parametros_cenario(cod, horizonte=h), seed)
            assert _resumo(r, h) == esperado, (
                f"{cod}, seed {seed}: {_resumo(r, h)} em vez de {esperado}. Se a versão do NumPy "
                "mudou, os fluxos aleatórios podem ter mudado; compare com analyses/conferir_rodada_anterior.py")
        # medir a percolação e guardar grades não altera a dinâmica
        r = simular(parametros_cenario(codigo, horizonte=h), seed, snapshots=(0, 50), percolacao=(0, 10, 50, h))
        assert _resumo(r, h) == esperado


def test_12_infectados_espalhados():
    """Condição E: 9 sítios sorteados, iguais entre cenários, diferentes entre seeds, mesma população de C."""
    a = Simulacao(parametros_cenario("F1E"), 0)
    b = Simulacao(parametros_cenario("M6E"), 0)
    c = Simulacao(parametros_cenario("F1E"), 1)
    central = Simulacao(parametros_cenario("F1C"), 0)
    assert a.infectado.sum() == 9 and a.n_iniciais == 9
    assert np.array_equal(a.infectado, b.infectado)            # mesma seed, cenários diferentes
    assert not np.array_equal(a.infectado, c.infectado)        # seeds diferentes
    assert np.array_equal(a.S0, central.S0) and np.array_equal(a.T0, central.T0)
    # bloco central: quadrado 3 x 3 no meio da grade
    g = central.infectado.reshape(100, 100)
    assert g.sum() == 9 and g[49:52, 49:52].all()
    # espalhados de verdade: em 200 seeds, os sítios sorteados cobrem a grade sem preferência pelo centro
    linhas = np.concatenate([np.flatnonzero(Simulacao(parametros_cenario("F0E"), s).infectado) // 100
                             for s in range(200)])
    assert abs(linhas.mean() - 49.5) < 3 * 100 / np.sqrt(12 * linhas.size), linhas.mean()
    # varredura da semente
    for codigo, n in (("F0C_I1", 1), ("F0C_I25", 25), ("F0E_I25", 25), ("M0C_I81", 81), ("M0E_I81", 81)):
        sim = Simulacao(parametros_cenario(codigo), 2)
        assert sim.infectado.sum() == n, codigo
    g = Simulacao(parametros_cenario("M0C_I81"), 2).infectado.reshape(100, 100)
    assert g[46:55, 46:55].all()
    # a dinâmica roda e conserva: toda infecção tem um transmissor
    r = simular(parametros_cenario("F2E", horizonte=150), 4)
    assert r["novas_inf_faixa"].sum() == r["trans_faixa"].sum() > 0


def test_13_estados():
    """Os quatro estados somam a população e batem com infectados, vacinados e recuperados."""
    for codigo in ("F0C", "F1E", "F5C", "M0E", "M1C", "M6E"):
        p = parametros_cenario(codigo, horizonte=300)
        sim = Simulacao(p, 1)
        r = sim.rodar(snapshots=(0, 10, 100, 300))
        ne = r["n_estado"].astype(int)
        assert (ne.sum(axis=1) == p.N ** 2).all(), codigo
        assert np.array_equal(ne[:, INFECTADO], r["n_inf_faixa"].sum(axis=1)), codigo
        for j, t in enumerate(r["grades_tempos"]):
            g = r["grades_estado"][j]
            assert np.array_equal(np.bincount(g.ravel(), minlength=4), ne[t]), (codigo, t)
            assert np.array_equal(g == INFECTADO, r["grades_infectado"][j])
            # suscetível = não infectado com m abaixo do limiar
            assert np.array_equal(g == SUSCETIVEL, (r["grades_m"][j] < 0.5) & ~r["grades_infectado"][j])
        if p.regime == "nenhum":
            assert ne[:, VACINADO].sum() == 0, codigo
        if codigo.startswith("M0"):
            # SIR: recuperados = infecções acumuladas - infectados; suscetíveis não aumentam
            acumuladas = 9 + np.concatenate([[0], np.cumsum(r["novas_inf_faixa"].sum(axis=1))[:-1]])
            assert np.array_equal(ne[:, RECUPERADO], acumuladas - ne[:, INFECTADO])
            assert (np.diff(ne[:, SUSCETIVEL]) <= 0).all()
    # vacinação em t = 0 no Measles-like: em t = 10 metade dos sítios já está no estado vacinado
    r = simular(parametros_cenario("M1C", horizonte=20), 0)
    assert abs(int(r["n_estado"][10, VACINADO]) - 4996) <= 15
    # primeiras doses em recuperados: nenhuma em t = 0, várias em t = 60
    assert simular(parametros_cenario("F1C", horizonte=100), 0)["doses1_em_recuperados"] == 0
    assert simular(parametros_cenario("F3C", horizonte=100), 0)["doses1_em_recuperados"] > 0


def _aglomerados_forca_bruta(mask):
    """Busca com coordenadas desdobradas: lista ordenada de (tamanho, envolve o toro)."""
    N = mask.shape[0]
    onde = {}
    saida = []
    for i0 in range(N):
        for j0 in range(N):
            if not mask[i0, j0] or (i0, j0) in onde:
                continue
            onde[(i0, j0)] = (i0, j0)
            pilha, tam, env = [(i0, j0)], 0, False
            while pilha:
                i, j = pilha.pop()
                ui, uj = onde[(i, j)]
                tam += 1
                for di, dj in DESLOCAMENTOS:
                    viz = ((i + di) % N, (j + dj) % N)
                    if not mask[viz]:
                        continue
                    if viz in onde:
                        env = env or onde[viz] != (ui + di, uj + dj)
                    else:
                        onde[viz] = (ui + di, uj + dj)
                        pilha.append(viz)
            saida.append((tam, env))
    return sorted(saida)


def test_14_percolacao():
    """Aglomerados periódicos com vizinhança de Moore: casos construídos e comparação com força bruta."""
    N = 10
    linha = np.zeros((N, N), bool); linha[3, :] = True
    diagonal = np.zeros((N, N), bool); diagonal[np.arange(N), np.arange(N)] = True
    canto = np.zeros((N, N), bool); canto[0, 0] = canto[0, N - 1] = canto[N - 1, 0] = True
    bloco = np.zeros((N, N), bool); bloco[2:5, 2:5] = True
    for mask, tam, env in ((linha, [10], [True]), (diagonal, [10], [True]), (canto, [3], [False]),
                           (bloco, [9], [False]), (np.ones((N, N), bool), [100], [True]),
                           (np.zeros((N, N), bool), [], [])):
        _, t, e = aglomerados_periodicos(mask)
        assert t.tolist() == tam and e.tolist() == env
    rng = np.random.default_rng(1)
    for N in (6, 13, 24):
        for p in (0.2, 0.35, 0.41, 0.5, 0.7):
            for _ in range(15):
                mask = rng.random((N, N)) < p
                rot, t, e = aglomerados_periodicos(mask)
                assert sorted(zip(t.tolist(), e.tolist())) == _aglomerados_forca_bruta(mask), (N, p)
                assert np.array_equal(rot > 0, mask) and t.sum() == mask.sum()
    med = medidas_percolacao(bloco)
    assert med["p"] == 0.09 and med["p_inf"] == 0.09 and med["n_aglom"] == 1 and not med["envolve"]
    assert med["hist"].sum() == 1 and med["hist"][3] == 1          # tamanho 9 está em [8, 16)
    # acima do limiar (0,407) o maior aglomerado envolve o toro; bem abaixo, não
    alto = [medidas_percolacao(rng.random((100, 100)) < 0.55)["envolve"] for _ in range(10)]
    baixo = [medidas_percolacao(rng.random((100, 100)) < 0.25)["envolve"] for _ in range(10)]
    assert all(alto) and not any(baixo)
    # dentro da simulação: em t = 0 quase todos são suscetíveis e o aglomerado envolve o toro
    r = simular(parametros_cenario("M1C", horizonte=60), 0, percolacao=(0, 10, 60))
    assert r["perc_tempos"].tolist() == [0, 10, 60]
    assert r["perc_p"][0] > 0.99 and r["perc_envolve"][0]
    assert abs(r["perc_p"][1] - 0.5) < 0.01                        # metade vacinada, já protegida em t = 10
    assert np.isclose(r["perc_p"][2], r["n_estado"][60, SUSCETIVEL] / 1e4)


def test_15_configuracoes():
    """52 configurações: 30 principais, 10 de semente, 12 de conexões, com os parâmetros certos."""
    assert len(CONFIGURACOES) == 52
    assert len(codigos_do_grupo("principal")) == 30
    assert len(codigos_do_grupo("semente")) == 10 and len(codigos_do_grupo("conexoes")) == 12
    for base in CENARIOS:
        pb, pc, pe = (parametros_cenario(base + x) for x in ("", "C", "E"))
        assert pb == pc
        assert pe.init_espalhados == 9 and pc.init_espalhados == 0 and pc.init_cluster == 3
        # C e E só diferem na condição inicial
        assert {k for k in pc.__dataclass_fields__ if getattr(pc, k) != getattr(pe, k)} == {"init_espalhados"}
    for codigo, v in CONFIGURACOES.items():
        p = parametros_cenario(codigo)
        if v["grupo"] != "principal":
            assert p.regime == "nenhum", codigo              # varreduras só sem vacina
        if v["grupo"] == "conexoes":
            assert p.K_encontros in (0, 1, 10) and (p.init_espalhados == 9 or p.init_cluster == 3)
        else:
            assert p.K_encontros == 3
    assert parametros_cenario("M0E_K10").K_encontros == 10 and parametros_cenario("F0C_K0").K_encontros == 0
    # K = 0 (só vizinhos) roda e atribui transmissores
    r = simular(parametros_cenario("M0C_K0", horizonte=80), 0)
    assert r["novas_inf_faixa"].sum() == r["trans_faixa"].sum() > 0


TESTES = [test_01_contribuicoes_locais, test_02_equivalencia_com_modelo_atual, test_03_conservacao,
          test_04_curva_de_bateman, test_05_regra_de_reforco, test_06_doses_iguais,
          test_07_alocacao_gaussiana, test_08_faixa_protegida, test_09_measles_sem_reinfeccao,
          test_10_reprodutibilidade, test_11_regressao_bloco_central, test_12_infectados_espalhados,
          test_13_estados, test_14_percolacao, test_15_configuracoes]


def main(argv):
    escolhidos = {int(a) for a in argv} if argv else None
    falhas = 0
    for i, teste in enumerate(TESTES, start=1):
        if escolhidos and i not in escolhidos:
            continue
        t0 = time.perf_counter()
        try:
            teste()
            print(f"[PASSOU] {teste.__name__} ({time.perf_counter() - t0:.1f} s): {teste.__doc__}")
        except Exception:
            falhas += 1
            print(f"[FALHOU] {teste.__name__} ({time.perf_counter() - t0:.1f} s): {teste.__doc__}")
            traceback.print_exc()
    print(f"\n{falhas} teste(s) falharam." if falhas else "\nTodos os testes executados passaram.")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
