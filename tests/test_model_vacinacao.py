"""
Testes do modelo de vacinação (seção 8 da especificação).

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
from src.model_vacinacao import (DESLOCAMENTOS, Simulacao, contribuicoes_locais, curva_dose,
                                 indices_vizinhos, instante_pico, mascara_passos, parametros_cenario,
                                 simular, taxas_por_faixa)


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
                     "grades_seeds", "grades_tempos"):
                continue
            if k.startswith("grades_"):   # só a seed 0 guarda grades
                assert np.array_equal(juntas[k], so0[k]), (codigo, k)
                continue
            assert np.array_equal(juntas[k][0], so0[k][0]), (codigo, k)
            assert np.array_equal(juntas[k][1], so1[k][0]), (codigo, k)

    # a seed k produz os mesmos S0 e T0 em todos os cenários
    s_f = Simulacao(parametros_cenario("F2"), 7)
    s_m = Simulacao(parametros_cenario("M5"), 7)
    assert np.array_equal(s_f.S0, s_m.S0) and np.array_equal(s_f.T0, s_m.T0)


TESTES = [test_01_contribuicoes_locais, test_02_equivalencia_com_modelo_atual, test_03_conservacao,
          test_04_curva_de_bateman, test_05_regra_de_reforco, test_06_doses_iguais,
          test_07_alocacao_gaussiana, test_08_faixa_protegida, test_09_measles_sem_reinfeccao,
          test_10_reprodutibilidade]


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
