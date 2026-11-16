"""
Modelo small-world com imunidade que cresce e decai, doses de reforço,
alocação de vacinas por suscetibilidade e atribuição de cada infecção ao seu
transmissor.

A dinâmica de contágio é a mesma de ``src/model.py``: grade N x N periódica,
8 vizinhos de Moore (peso ``W_local``) mais ``K_encontros`` encontros de
mercado sorteados a cada passo (peso ``W_mercado``), atualização síncrona.
O que muda:

* o estado de cada sítio é explícito (arrays ``infectado``, ``n_doses``,
  instantes das duas doses mais recentes, instante da última recuperação),
  em vez de codificado pelos zeros de S e T;
* cada indivíduo tem uma imunidade ``m`` em [0, 1] que reduz a suscetibilidade
  e a transmissibilidade: ``S_ef = S0 (1 - m)`` e ``T_ef = T0 (1 - m)``;
* ``m = max(m_vac, m_nat)``; a parte vacinal segue uma curva de Bateman por
  dose e a parte natural decai exponencialmente depois da recuperação;
* cada infecção é atribuída a um transmissor por riscos competitivos, o que
  permite contar infecções e transmissões por faixa de imunidade;
* a semente aleatória é um parâmetro e gera quatro fluxos independentes
  (atributos S0/T0, dinâmica, sorteio de vacinados, posição dos infectados
  iniciais espalhados);
* os infectados iniciais ficam em um bloco central ou espalhados pela grade;
* cada sítio é classificado em quatro estados (suscetível, infectado,
  recuperado, vacinado), e os aglomerados de suscetíveis são medidos como em
  um problema de percolação de sítios.

Uso mínimo, a partir da raiz do repositório::

    from src.model_vacinacao import parametros_cenario, simular
    res = simular(parametros_cenario("F1", horizonte=400), seed=0)
    I = res["n_inf_faixa"].sum(axis=1)          # infectados em cada passo
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

# ---------------------------------------------------------------------------
# Faixas de status (seção 5 da especificação)
# ---------------------------------------------------------------------------
N_FAIXAS = 5
NOMES_FAIXAS = (
    "Nunca vacinado",
    "Vacinado, m ≥ 0,90",
    "Vacinado, 0,75 ≤ m < 0,90",
    "Vacinado, 0,50 ≤ m < 0,75",
    "Vacinado, m < 0,50",
)
# O histograma de infecções por indivíduo vai de 0 a MAX_HIST; a última
# posição acumula "MAX_HIST ou mais".
MAX_HIST = 40

# Estados categóricos de um sítio (grades e série ``n_estado``):
#   0 suscetível: não infectado e com imunidade m abaixo de ``limiar_suscetivel``;
#   1 infectado;
#   2 recuperado: não infectado, m acima do limiar e imunidade natural >= vacinal;
#   3 vacinado:   não infectado, m acima do limiar e imunidade vacinal > natural.
N_ESTADOS = 4
NOMES_ESTADOS = ("Suscetível", "Infectado", "Recuperado", "Vacinado")
SUSCETIVEL, INFECTADO, RECUPERADO, VACINADO = range(4)

# Limiar de percolação de sítios na rede quadrada com vizinhança de Moore
# (primeiros e segundos vizinhos). Essa rede é a rede "matching" da rede
# quadrada, então p_c = 1 - p_c(quadrada) = 1 - 0,592746 (Sykes e Essam, 1964).
P_C_MOORE = 0.407254
# Histograma de tamanhos de aglomerado em classes [1, 2), [2, 4), [4, 8), ...
N_CLASSES_AGLOM = 15

# Deslocamentos (di, dj) dos 8 vizinhos de Moore, na ordem usada em todo o módulo.
DESLOCAMENTOS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


# ---------------------------------------------------------------------------
# Parâmetros
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Parametros:
    """Todos os parâmetros do modelo. Os padrões são os do cenário Flu-like sem vacina."""

    # --- dinâmica de contágio (iguais aos de src/model.py) ---
    N: int = 100                 # lado da grade
    alpha: float = 0.08          # constante de acoplamento da infecção
    p_rec: float = 0.05          # probabilidade de recuperação por passo
    W_local: float = 1.0         # peso de cada vizinho de Moore
    W_mercado: float = 0.4       # peso de cada encontro de mercado
    K_encontros: int = 3         # encontros de mercado por indivíduo por passo
    init_cluster: int = 3        # lado do bloco central de infectados iniciais
    init_espalhados: int = 0     # se > 0: este número de infectados iniciais em sítios sorteados (ignora o bloco)
    atrib_min: float = 0.2       # S0 e T0 ~ U(atrib_min, atrib_max)
    atrib_max: float = 1.0
    horizonte: int = 1500        # número de passos; as séries têm horizonte + 1 pontos

    # --- imunidade (seção 3) ---
    tau_r: float = 3.0           # subida da curva vacinal
    tau_d1: float = 100.0        # decaimento da primeira dose (inf = sem decaimento)
    tau_d2: float = 150.0        # decaimento da segunda dose em diante
    tau_n: float = 100.0         # decaimento da imunidade natural (inf = permanente)
    reforco: bool = True         # doses de reforço habilitadas
    tau_b: float = 20.0          # escala da probabilidade de reforço
    limiar_reforco: float = 0.5  # reforço só com m abaixo deste valor, depois do pico

    # --- primeira dose (seção 4) ---
    regime: str = "nenhum"       # "nenhum", "instantaneo" ou "continuo"
    t_vac: int = 0               # passo da aplicação instantânea ou início do regime contínuo
    alocacao: str = "aleatoria"  # "aleatoria" ou "gaussiana"
    cobertura: float = 0.5       # fração dos elegíveis (instantâneo) ou da população (contínuo)
    doses_por_passo: int = 50    # regime contínuo
    sigma: float = 0.2           # largura do peso gaussiano em S0
    centro_gauss: float = 1.0    # centro do peso gaussiano (máximo de S0)

    # --- métricas e execução ---
    limites_faixas: tuple = (0.5, 0.75, 0.9)
    limiar_suscetivel: float = 0.5   # m abaixo disto (e sem infecção) = estado suscetível
    espera_parada: int = 30      # passos sem doses antes de permitir a parada antecipada

    def __post_init__(self):
        if self.regime not in ("nenhum", "instantaneo", "continuo"):
            raise ValueError(f"regime desconhecido: {self.regime!r}")
        if self.alocacao not in ("aleatoria", "gaussiana"):
            raise ValueError(f"alocação desconhecida: {self.alocacao!r}")
        if not (self.tau_r < self.tau_d1 and self.tau_r < self.tau_d2):
            raise ValueError("a curva de Bateman exige tau_r < tau_d")
        if self.init_espalhados < 0 or self.init_espalhados > self.N * self.N:
            raise ValueError("init_espalhados deve estar entre 0 e N*N")
        if len(self.limites_faixas) != 3 or list(self.limites_faixas) != sorted(self.limites_faixas):
            raise ValueError("limites_faixas deve ter 3 valores crescentes")

    def para_json(self) -> str:
        """Parâmetros como texto JSON (infinito aparece como ``Infinity``)."""
        return json.dumps(asdict(self), ensure_ascii=False)


DOENCAS = {
    # imunidade vacinal e natural que decaem, com reforços
    "flu": dict(tau_d1=100.0, tau_d2=150.0, tau_n=100.0, reforco=True),
    # imunidade duradoura: sem decaimento e sem reforço
    "measles": dict(tau_d1=np.inf, tau_d2=np.inf, tau_n=np.inf, reforco=False),
}

_T0 = dict(regime="instantaneo", t_vac=0)
_T60 = dict(regime="instantaneo", t_vac=60)
_CONT = dict(regime="continuo", t_vac=0)

# Os 15 cenários da seção 6. "rotulo" é usado só nas figuras.
CENARIOS = {
    "F0": dict(doenca="flu", regime="nenhum"),
    "F1": dict(doenca="flu", alocacao="aleatoria", **_T0),
    "F2": dict(doenca="flu", alocacao="gaussiana", **_T0),
    "F3": dict(doenca="flu", alocacao="aleatoria", **_T60),
    "F4": dict(doenca="flu", alocacao="gaussiana", **_T60),
    "F5": dict(doenca="flu", alocacao="aleatoria", **_CONT),
    "F6": dict(doenca="flu", alocacao="gaussiana", **_CONT),
    "F7": dict(doenca="flu", alocacao="aleatoria", reforco=False, **_T0),
    "M0": dict(doenca="measles", regime="nenhum"),
    "M1": dict(doenca="measles", alocacao="aleatoria", **_T0),
    "M2": dict(doenca="measles", alocacao="gaussiana", **_T0),
    "M3": dict(doenca="measles", alocacao="aleatoria", **_T60),
    "M4": dict(doenca="measles", alocacao="gaussiana", **_T60),
    "M5": dict(doenca="measles", alocacao="aleatoria", **_CONT),
    "M6": dict(doenca="measles", alocacao="gaussiana", **_CONT),
}


# Condição inicial: "C" = 9 infectados em um bloco 3 x 3 no centro da grade;
# "E" = 9 infectados em sítios sorteados (espalhados).
CONDICOES = {"C": "bloco central", "E": "espalhados"}
N_INICIAIS = 9
LADOS_SEMENTE = (1, 5, 9)        # varredura da semente: I0 = 1, 25 e 81 (I0 = 9 é o caso principal)
K_VARREDURA = (0, 1, 10)         # varredura de encontros de mercado (K = 3 é o caso principal)


def _montar_configuracoes() -> dict:
    """Todas as configurações do ensemble, na ordem usada pelo job array.

    Cada entrada tem ``base`` (um dos 15 cenários), ``grupo``, ``condicao``
    ("C" ou "E") e ``extras`` (campos de Parametros que mudam em relação ao cenário).

    * ``principal``: os 15 cenários nas duas condições iniciais (``F1C``, ``F1E``...);
    * ``semente``: F0 e M0 com I0 = 1, 25 e 81 (``F0C_I25``, ``F0E_I25``...). Com
      I0 = 1 só existe a versão C: um único infectado em uma grade periódica é
      estatisticamente a mesma coisa no centro ou em um sítio sorteado;
    * ``conexoes``: F0 e M0 com K = 0, 1 e 10 encontros de mercado (``F0C_K0``...).
    """
    conf = {}
    for cond, extras in (("C", {}), ("E", {"init_espalhados": N_INICIAIS})):
        for base in CENARIOS:
            conf[f"{base}{cond}"] = dict(base=base, grupo="principal", condicao=cond, extras=dict(extras))
    for base in ("F0", "M0"):
        for lado in LADOS_SEMENTE:
            n = lado * lado
            conf[f"{base}C_I{n}"] = dict(base=base, grupo="semente", condicao="C",
                                         extras=dict(init_cluster=lado))
            if n > 1:
                conf[f"{base}E_I{n}"] = dict(base=base, grupo="semente", condicao="E",
                                             extras=dict(init_espalhados=n))
    for base in ("F0", "M0"):
        for K in K_VARREDURA:
            for cond, extras in (("C", {}), ("E", {"init_espalhados": N_INICIAIS})):
                conf[f"{base}{cond}_K{K}"] = dict(base=base, grupo="conexoes", condicao=cond,
                                                  extras=dict(extras, K_encontros=K))
    return conf


CONFIGURACOES = _montar_configuracoes()
GRUPOS = ("principal", "semente", "conexoes")


def codigos_do_grupo(grupo: str) -> list:
    """Códigos das configurações de um grupo, na ordem de ``CONFIGURACOES``."""
    return [c for c, v in CONFIGURACOES.items() if v["grupo"] == grupo]


def parametros_cenario(codigo: str, **extras) -> Parametros:
    """Parâmetros de um cenário ou de uma configuração do ensemble.

    Aceita os 15 códigos de cenário (``"F0"`` a ``"F7"``, ``"M0"`` a ``"M6"``; a
    condição inicial é o bloco central, então ``"F1"`` equivale a ``"F1C"``) e
    os códigos de ``CONFIGURACOES`` (``"F1E"``, ``"F0C_I25"``, ``"M0E_K10"``...).
    ``extras`` sobrescreve qualquer campo, por exemplo ``horizonte=400``.
    """
    if codigo in CENARIOS:
        base_cod, extras_conf = codigo, {}
    elif codigo in CONFIGURACOES:
        base_cod, extras_conf = CONFIGURACOES[codigo]["base"], CONFIGURACOES[codigo]["extras"]
    else:
        raise KeyError(f"cenário desconhecido: {codigo!r}; opções: {', '.join(CENARIOS)} "
                       f"ou as configurações {', '.join(CONFIGURACOES)}")
    campos = dict(CENARIOS[base_cod])
    doenca = campos.pop("doenca")
    base = dict(DOENCAS[doenca])
    base.update(campos)
    base.update(extras_conf)
    base.update(extras)
    return Parametros(**base)


# ---------------------------------------------------------------------------
# Curva de imunidade de uma dose (função de Bateman)
# ---------------------------------------------------------------------------
def instante_pico(tau_r: float, tau_d: float) -> float:
    """Tempo desde a dose em que a curva atinge o máximo; infinito se ``tau_d`` é infinito."""
    if np.isinf(tau_d):
        return np.inf
    return tau_r * tau_d / (tau_d - tau_r) * np.log(tau_d / tau_r)


def norma_bateman(tau_r: float, tau_d: float) -> float:
    """Constante A que faz o pico da curva valer exatamente 1."""
    if np.isinf(tau_d):
        return 1.0
    ts = instante_pico(tau_r, tau_d)
    return 1.0 / (np.exp(-ts / tau_d) - np.exp(-ts / tau_r))


def curva_dose(tau, tau_r: float, tau_d: float):
    """Imunidade gerada por uma dose, ``tau`` passos depois dela.

    ``b(tau) = A (exp(-tau/tau_d) - exp(-tau/tau_r))`` para ``tau >= 0`` e 0 antes.
    Com ``tau_d`` infinito a curva vira ``1 - exp(-tau/tau_r)``.
    """
    tau = np.asarray(tau, dtype=float)
    A = norma_bateman(tau_r, tau_d)
    b = A * (np.exp(-tau / tau_d) - np.exp(-tau / tau_r))
    return np.where(tau >= 0, np.clip(b, 0.0, 1.0), 0.0)


# ---------------------------------------------------------------------------
# Contribuições locais
# ---------------------------------------------------------------------------
def contribuicoes_locais(T2d: np.ndarray) -> np.ndarray:
    """Empilha, para cada sítio, o valor de ``T2d`` em cada um dos 8 vizinhos de Moore.

    Devolve um array (8, N, N) em que ``saida[k, i, j] = T2d[(i+di) % N, (j+dj) % N]``
    com ``(di, dj) = DESLOCAMENTOS[k]``. A soma ao longo do eixo 0 é igual a
    ``convolve2d(T2d, kernel, mode='same', boundary='wrap')`` do modelo original.
    """
    return np.stack([np.roll(T2d, shift=(-di, -dj), axis=(0, 1)) for di, dj in DESLOCAMENTOS])


def indices_vizinhos(N: int) -> np.ndarray:
    """Índices lineares dos 8 vizinhos de cada sítio: array (8, N*N), mesma ordem de DESLOCAMENTOS."""
    i, j = np.divmod(np.arange(N * N), N)
    return np.stack([((i + di) % N) * N + (j + dj) % N for di, dj in DESLOCAMENTOS]).astype(np.int64)


# ---------------------------------------------------------------------------
# Aglomerados de suscetíveis (percolação de sítios, vizinhança de Moore)
# ---------------------------------------------------------------------------
_MOORE = np.ones((3, 3), dtype=int)


def aglomerados_periodicos(mask: np.ndarray):
    """Aglomerados de ``mask`` (N x N, booleano) com vizinhança de Moore e contorno periódico.

    Devolve ``(rotulos, tamanhos, envolve)``: ``rotulos`` é um array N x N com 0
    fora da máscara e 1..n em cada aglomerado; ``tamanhos[k - 1]`` é o número de
    sítios do aglomerado k; ``envolve[k - 1]`` diz se ele dá a volta no toro
    (existe um caminho fechado dentro dele que não pode ser contraído a um ponto).

    Método: rotula com contorno aberto e depois une os rótulos pelas ligações
    que atravessam as bordas. Cada ligação carrega o deslocamento (em número de
    voltas) entre as duas cópias periódicas; um aglomerado envolve o toro quando
    duas ligações o fecham com deslocamentos incompatíveis (união com pesos).
    """
    mask = np.asarray(mask, dtype=bool)
    N = mask.shape[0]
    rot, n = ndimage.label(mask, structure=_MOORE)
    if n == 0:
        return rot, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)

    # Ligações que cruzam a borda: (a, b, wi, wj) com cópia(b) = cópia(a) + (wi, wj).
    idx = np.arange(N)
    lig = []
    for d in (-1, 0, 1):
        k2 = idx + d
        w = np.floor_divide(k2, N)          # -1, 0 ou +1: volta na outra direção (cantos)
        k2 = k2 % N
        um = np.ones(N, dtype=np.int64)
        lig.append(np.stack([rot[idx, N - 1], rot[k2, 0], w, um], axis=1))    # cruza a última coluna
        lig.append(np.stack([rot[N - 1, idx], rot[0, k2], um, w], axis=1))    # cruza a última linha
    lig = np.concatenate(lig)
    lig = np.unique(lig[(lig[:, 0] > 0) & (lig[:, 1] > 0)], axis=0)

    pai = list(range(n + 1))
    des = [(0, 0)] * (n + 1)       # cópia(nó) - cópia(pai do nó)
    volta = [False] * (n + 1)      # válido nas raízes

    def achar(a):
        caminho = []
        while pai[a] != a:
            caminho.append(a)
            a = pai[a]
        ai = aj = 0
        for x in reversed(caminho):           # compressão de caminho, acumulando deslocamentos
            ai, aj = ai + des[x][0], aj + des[x][1]
            pai[x], des[x] = a, (ai, aj)
        return a

    for a, b, wi, wj in lig.tolist():
        ra, rb = achar(a), achar(b)
        oa = des[a] if a != ra else (0, 0)
        ob = des[b] if b != rb else (0, 0)
        if ra == rb:
            if (ob[0] - oa[0], ob[1] - oa[1]) != (wi, wj):
                volta[ra] = True
        else:
            pai[rb] = ra
            des[rb] = (wi + oa[0] - ob[0], wj + oa[1] - ob[1])
            volta[ra] = volta[ra] or volta[rb]

    raiz = np.array([0] + [achar(k) for k in range(1, n + 1)])
    raizes, novo = np.unique(raiz[1:], return_inverse=True)
    mapa = np.concatenate([[0], novo + 1])
    rotulos = mapa[rot]
    tamanhos = np.bincount(rotulos.ravel(), minlength=raizes.size + 1)[1:]
    envolve = np.array([volta[r] for r in raizes], dtype=bool)
    return rotulos, tamanhos, envolve


def medidas_percolacao(mask: np.ndarray) -> dict:
    """Medidas de percolação dos sítios de ``mask`` (os suscetíveis).

    * ``p``: fração de sítios na máscara (parâmetro de controle);
    * ``p_inf``: fração de todos os sítios que está no maior aglomerado (parâmetro de ordem);
    * ``n_aglom``: número de aglomerados;
    * ``envolve``: o maior aglomerado dá a volta no toro (critério de percolação em rede finita);
    * ``s_medio``: tamanho médio dos aglomerados finitos, sum(s^2) / sum(s) sem o maior
      (análogo da suscetibilidade; tem máximo perto do limiar);
    * ``hist``: número de aglomerados com tamanho em [1, 2), [2, 4), [4, 8), ...
    """
    mask = np.asarray(mask, dtype=bool)
    _, tam, env = aglomerados_periodicos(mask)
    hist = np.zeros(N_CLASSES_AGLOM, dtype=np.int32)
    if tam.size == 0:
        return dict(p=0.0, p_inf=0.0, n_aglom=0, envolve=False, s_medio=0.0, hist=hist)
    classes = np.minimum(np.floor(np.log2(tam)).astype(int), N_CLASSES_AGLOM - 1)
    hist += np.bincount(classes, minlength=N_CLASSES_AGLOM).astype(np.int32)
    maior = int(np.argmax(tam))
    resto = np.delete(tam, maior).astype(float)
    s_medio = float((resto ** 2).sum() / resto.sum()) if resto.size else 0.0
    return dict(p=float(mask.mean()), p_inf=float(tam[maior] / mask.size), n_aglom=int(tam.size),
                envolve=bool(env[maior]), s_medio=s_medio, hist=hist)


# ---------------------------------------------------------------------------
# Simulação
# ---------------------------------------------------------------------------
class Simulacao:
    """Uma realização do modelo. Use :func:`simular` para o caso comum.

    A classe expõe o estado (``infectado``, ``n_doses``, ``S0``...) para que
    os testes possam inspecioná-lo depois de :meth:`rodar`.
    """

    def __init__(self, p: Parametros, seed: int):
        self.p = p
        self.seed = int(seed)
        N2 = p.N * p.N

        # Quatro fluxos independentes derivados da seed: atributos, dinâmica,
        # vacinação e posição dos infectados iniciais espalhados. A seed k
        # produz os mesmos S0 e T0 em todos os cenários, e a dinâmica não
        # consome números do fluxo de vacinação (comparações pareadas). Os três
        # primeiros fluxos são os mesmos de quando só havia três: spawn(4)
        # começa pelos mesmos filhos de spawn(3).
        ss_attr, ss_din, ss_vac, ss_ini = np.random.SeedSequence(self.seed).spawn(4)
        rng_attr = np.random.default_rng(ss_attr)
        self.rng_din = np.random.default_rng(ss_din)
        self.rng_vac = np.random.default_rng(ss_vac)

        # Atributos intrínsecos, fixos durante toda a simulação.
        self.S0 = rng_attr.uniform(p.atrib_min, p.atrib_max, N2)
        self.T0 = rng_attr.uniform(p.atrib_min, p.atrib_max, N2)

        # --- estado explícito por sítio ---
        self.infectado = np.zeros(N2, dtype=bool)
        self.n_doses = np.zeros(N2, dtype=np.int32)
        # duas doses mais recentes: instante e índice do decaimento (0 = tau_d1, 1 = tau_d2)
        self.t_dose_rec = np.full(N2, np.nan)   # mais recente
        self.k_dose_rec = np.zeros(N2, dtype=np.int8)
        self.t_dose_ant = np.full(N2, np.nan)   # penúltima
        self.k_dose_ant = np.zeros(N2, dtype=np.int8)
        self.t_rec = np.full(N2, np.nan)        # instante da última recuperação
        self.t_cruzou = np.full(N2, np.nan)     # instante em que m cruzou o limiar descendo
        self.n_infec = np.zeros(N2, dtype=np.int32)  # infecções sofridas por cada indivíduo

        # Infectados iniciais, como em src/model.py: sítios sorteados sem
        # reposição (init_espalhados > 0) ou um bloco central.
        grade = self.infectado.reshape(p.N, p.N)
        if p.init_espalhados > 0:
            # As posições dependem só da seed e do número de infectados: a seed k
            # tem os mesmos infectados iniciais em todos os cenários.
            sorteados = np.random.default_rng(ss_ini).choice(N2, size=p.init_espalhados, replace=False)
            self.infectado[sorteados] = True
        else:
            c, r = p.N // 2, p.init_cluster // 2
            if p.init_cluster == 1:
                grade[c, c] = True
            else:
                grade[c - r:c + r + 1, c - r:c + r + 1] = True
        self.n_infec[self.infectado] = 1
        self.n_iniciais = int(self.infectado.sum())

        # Tabelas da curva de Bateman para os dois decaimentos.
        self._taud = np.array([p.tau_d1, p.tau_d2], dtype=float)
        self._A = np.array([norma_bateman(p.tau_r, td) for td in self._taud])
        self._tpico = np.array([instante_pico(p.tau_r, td) for td in self._taud])

        self._vizinhos = indices_vizinhos(p.N)
        # logaritmo do peso gaussiano em S0 (evita underflow com sigma pequeno)
        self._log_peso_gauss = -(self.S0 - p.centro_gauss) ** 2 / (2 * p.sigma ** 2)

        # Controle da primeira dose.
        self._total_continuo = int(np.floor(p.cobertura * N2 + 0.5))
        self._doses1_aplicadas = 0
        self.doses1_em_recuperados = 0   # primeiras doses dadas a quem já teve a doença
        self._inst_aplicada = False
        self._t_ultima_dose = -np.inf

        # Sem decaimento algum o estado congela quando não há infectados nem
        # doses por aplicar; só então a parada antecipada é permitida.
        self._congela = bool(np.isinf(p.tau_d1) and np.isinf(p.tau_d2) and np.isinf(p.tau_n))

        # --- séries (horizonte + 1 pontos) ---
        T1 = p.horizonte + 1
        self.n_inf_faixa = np.zeros((T1, N_FAIXAS), dtype=np.int32)      # infectados por faixa
        self.n_naoinf_faixa = np.zeros((T1, N_FAIXAS), dtype=np.int32)   # não infectados por faixa
        self.novas_inf_faixa = np.zeros((T1, N_FAIXAS), dtype=np.int32)  # por faixa do infectado
        self.trans_faixa = np.zeros((T1, N_FAIXAS), dtype=np.int32)      # por faixa do transmissor
        self.doses1 = np.zeros(T1, dtype=np.int32)
        self.reforcos = np.zeros(T1, dtype=np.int32)
        self.imunidade_media = np.zeros(T1, dtype=np.float32)
        # sítios em cada estado (suscetível, infectado, recuperado, vacinado)
        self.n_estado = np.zeros((T1, N_ESTADOS), dtype=np.int16 if N2 <= 32767 else np.int32)
        self.t_extincao = -1   # primeiro passo sem infectados (-1 = nunca)
        self.t_parada = p.horizonte  # último passo efetivamente simulado
        self.auditoria_reforcos = []  # preenchida só com auditoria=True

    # ----- imunidade ---------------------------------------------------------
    def imunidade(self, t: int):
        """Devolve ``(m, m_vac, m_nat)`` no passo ``t`` para todos os sítios."""
        p = self.p
        N2 = self.S0.size
        m_vac = np.zeros(N2)
        v = np.flatnonzero(self.n_doses > 0)
        if v.size:
            k = self.k_dose_rec[v]
            tau = t - self.t_dose_rec[v]
            curva = self._A[k] * (np.exp(-tau / self._taud[k]) - np.exp(-tau / p.tau_r))
            # penúltima dose: evita que um reforço derrube a proteção durante a subida
            dois = self.n_doses[v] >= 2
            if dois.any():
                k2 = self.k_dose_ant[v[dois]]
                tau2 = t - self.t_dose_ant[v[dois]]
                curva2 = self._A[k2] * (np.exp(-tau2 / self._taud[k2]) - np.exp(-tau2 / p.tau_r))
                curva[dois] = np.maximum(curva[dois], curva2)
            m_vac[v] = np.clip(curva, 0.0, 1.0)

        m_nat = np.zeros(N2)
        rec = np.flatnonzero(~np.isnan(self.t_rec))
        if rec.size:
            m_nat[rec] = np.exp(-(t - self.t_rec[rec]) / p.tau_n)  # tau_n infinito -> 1

        return np.maximum(m_vac, m_nat), m_vac, m_nat

    def faixas(self, m: np.ndarray) -> np.ndarray:
        """Faixa de status (0 a 4) de cada sítio, dada a imunidade ``m``."""
        nivel = np.digitize(m, self.p.limites_faixas)   # 0: m < 0,50 ... 3: m >= 0,90
        return np.where(self.n_doses > 0, 4 - nivel, 0)

    def estados(self, m: np.ndarray, m_vac: np.ndarray, m_nat: np.ndarray) -> np.ndarray:
        """Estado categórico (0 a 3) de cada sítio; veja ``NOMES_ESTADOS``."""
        imune = np.where(m_nat >= m_vac, RECUPERADO, VACINADO)
        estado = np.where(m < self.p.limiar_suscetivel, SUSCETIVEL, imune)
        return np.where(self.infectado, INFECTADO, estado).astype(np.int8)

    # ----- vacinação ---------------------------------------------------------
    def _sortear_primeira_dose(self, elegiveis: np.ndarray, n: int) -> np.ndarray:
        """Sorteia ``n`` elegíveis sem reposição, uniforme ou com peso gaussiano em S0."""
        if n <= 0:
            return elegiveis[:0]
        if self.p.alocacao == "aleatoria":
            return self.rng_vac.choice(elegiveis, size=n, replace=False)
        # Gumbel-top-k: equivale a sortear sem reposição, um por vez, com
        # probabilidade proporcional ao peso entre os que ainda restam.
        chaves = self._log_peso_gauss[elegiveis] + self.rng_vac.gumbel(size=elegiveis.size)
        return elegiveis[np.argpartition(chaves, -n)[-n:]]

    def _aplicar_dose(self, ids: np.ndarray, t: int) -> None:
        """Registra uma dose em ``t`` para os sítios ``ids`` (primeira dose ou reforço)."""
        self.t_dose_ant[ids] = self.t_dose_rec[ids]
        self.k_dose_ant[ids] = self.k_dose_rec[ids]
        self.t_dose_rec[ids] = t
        # primeira dose usa tau_d1; da segunda em diante, tau_d2
        self.k_dose_rec[ids] = np.where(self.n_doses[ids] == 0, 0, 1)
        self.n_doses[ids] += 1
        self.t_cruzou[ids] = np.nan
        self._t_ultima_dose = t

    def vacinacao_pendente(self) -> bool:
        """Ainda há primeiras doses por aplicar?"""
        p = self.p
        if p.regime == "instantaneo":
            return not self._inst_aplicada
        if p.regime == "continuo":
            return self._doses1_aplicadas < self._total_continuo
        return False

    def _vacinar(self, t: int, m: np.ndarray, auditoria: bool):
        """Aplica primeiras doses e reforços do passo ``t``. Devolve (n_primeiras, n_reforcos)."""
        p = self.p
        n1 = 0
        if p.regime == "instantaneo" and t == p.t_vac:
            elegiveis = np.flatnonzero((self.n_doses == 0) & ~self.infectado)
            n1 = int(np.floor(p.cobertura * elegiveis.size + 0.5))
            ids1 = self._sortear_primeira_dose(elegiveis, n1)
            self.doses1_em_recuperados += int((~np.isnan(self.t_rec[ids1])).sum())
            self._aplicar_dose(ids1, t)
            self._inst_aplicada = True
        elif p.regime == "continuo" and t >= p.t_vac and self._doses1_aplicadas < self._total_continuo:
            elegiveis = np.flatnonzero((self.n_doses == 0) & ~self.infectado)
            n1 = min(p.doses_por_passo, self._total_continuo - self._doses1_aplicadas, elegiveis.size)
            if n1 > 0:
                ids1 = self._sortear_primeira_dose(elegiveis, n1)
                self.doses1_em_recuperados += int((~np.isnan(self.t_rec[ids1])).sum())
                self._aplicar_dose(ids1, t)
        self._doses1_aplicadas += n1

        nref = 0
        if p.reforco and p.regime != "nenhum":
            v = np.flatnonzero(self.n_doses > 0)
            if v.size:
                tau = t - self.t_dose_rec[v]
                pos_pico = tau > self._tpico[self.k_dose_rec[v]]   # nunca durante a subida
                abaixo = pos_pico & (m[v] < p.limiar_reforco)
                tc = self.t_cruzou[v]
                tc[abaixo & np.isnan(tc)] = t       # acabou de cruzar o limiar descendo
                tc[~abaixo] = np.nan                # voltou a ficar acima (ou dose nova)
                self.t_cruzou[v] = tc
                cand = abaixo & ~self.infectado[v]
                if cand.any():
                    u = t - tc[cand]
                    p_b = 1.0 - np.exp(-u / p.tau_b)
                    toma = self.rng_vac.random(u.size) < p_b
                    ids = v[cand][toma]
                    nref = int(ids.size)
                    if nref:
                        if auditoria:
                            self.auditoria_reforcos.append(dict(
                                t=t, tau=tau[cand][toma].copy(), m=m[ids].copy(),
                                tpico=self._tpico[self.k_dose_rec[ids]].copy(),
                                infectado=self.infectado[ids].copy()))
                        self._aplicar_dose(ids, t)
        return n1, nref

    # ----- contágio ----------------------------------------------------------
    def _dinamica(self, t: int, m: np.ndarray, faixa: np.ndarray) -> None:
        """Infecções (com atribuição do transmissor) e recuperações do passo ``t``."""
        p = self.p
        N, N2, K = p.N, p.N * p.N, p.K_encontros
        inf = self.infectado
        T_ef = np.where(inf, self.T0 * (1.0 - m), 0.0)
        S_ef = np.where(inf, 0.0, self.S0 * (1.0 - m))

        locais = contribuicoes_locais(T_ef.reshape(N, N)).reshape(8, N2)
        pressao = p.W_local * locais.sum(axis=0)
        if K > 0:
            idx_mercado = self.rng_din.integers(0, N2, size=(N2, K))
            t_mercado = T_ef[idx_mercado]
            pressao = pressao + p.W_mercado * t_mercado.sum(axis=1)

        # mesma probabilidade do modelo original: 1 - exp(-alpha S_ef P)
        p_inf = -np.expm1(-p.alpha * S_ef * pressao)
        novos = np.flatnonzero(self.rng_din.random(N2) < p_inf)

        if novos.size:
            # Riscos competitivos: a fonte j é sorteada com probabilidade
            # proporcional à sua contribuição W_ij * T_ef_j.
            contrib = np.empty((novos.size, 8 + K))
            contrib[:, :8] = p.W_local * locais[:, novos].T
            if K > 0:
                contrib[:, 8:] = p.W_mercado * t_mercado[novos]
            acum = np.cumsum(contrib, axis=1)
            x = self.rng_din.random(novos.size) * acum[:, -1]
            col = np.minimum((acum <= x[:, None]).sum(axis=1), 8 + K - 1)
            fonte = self._vizinhos[np.minimum(col, 7), novos]
            if K > 0:
                de_mercado = col >= 8
                fonte[de_mercado] = idx_mercado[novos[de_mercado], col[de_mercado] - 8]
            self._ultima_contrib_fonte = contrib[np.arange(novos.size), col]   # para testes: sempre > 0
            self.novas_inf_faixa[t] = np.bincount(faixa[novos], minlength=N_FAIXAS)
            self.trans_faixa[t] = np.bincount(faixa[fonte], minlength=N_FAIXAS)

        recuperam = inf & (self.rng_din.random(N2) < p.p_rec)
        inf[recuperam] = False
        self.t_rec[recuperam] = t + 1      # primeiro passo já recuperado: m_nat = 1
        inf[novos] = True
        self.n_infec[novos] += 1

    # ----- laço principal ----------------------------------------------------
    def rodar(self, snapshots=(), percolacao=(), auditoria: bool = False) -> dict:
        """Roda até o horizonte e devolve as séries e os totais.

        ``snapshots``: passos em que as grades de ``m``, de infectados e de estados são guardadas.
        ``percolacao``: passos em que os aglomerados de suscetíveis são medidos.
        ``auditoria``: guarda, para cada reforço, o tempo desde a dose anterior,
        a imunidade e o estado de infecção (usado nos testes).
        """
        p = self.p
        N = p.N
        snapshots = sorted(set(int(s) for s in snapshots if 0 <= int(s) <= p.horizonte))
        percolacao = sorted(set(int(s) for s in percolacao if 0 <= int(s) <= p.horizonte))
        grades_m, grades_inf, grades_est, perc = {}, {}, {}, {}

        def registrar(s, m, estado):
            """Guarda as grades e as medidas de percolação do passo ``s``."""
            if s in snapshots:
                grades_m[s] = m.reshape(N, N).astype(np.float32)
                grades_inf[s] = self.infectado.reshape(N, N).copy()
                grades_est[s] = estado.reshape(N, N).copy()
            if s in percolacao:
                perc[s] = medidas_percolacao((estado == SUSCETIVEL).reshape(N, N))

        for t in range(p.horizonte + 1):
            m, m_vac, m_nat = self.imunidade(t)
            n1, nref = self._vacinar(t, m, auditoria)
            faixa = self.faixas(m)   # depois da vacinação: quem tomou dose agora já conta como vacinado
            estado = self.estados(m, m_vac, m_nat)

            inf = self.infectado
            self.n_inf_faixa[t] = np.bincount(faixa[inf], minlength=N_FAIXAS)
            self.n_naoinf_faixa[t] = np.bincount(faixa[~inf], minlength=N_FAIXAS)
            self.n_estado[t] = np.bincount(estado, minlength=N_ESTADOS)
            self.doses1[t] = n1
            self.reforcos[t] = nref
            self.imunidade_media[t] = m.mean()
            n_infectados = int(self.n_inf_faixa[t].sum())
            if n_infectados == 0 and self.t_extincao < 0:
                self.t_extincao = t
            registrar(t, m, estado)

            if t == p.horizonte:
                break

            if n_infectados == 0:
                # Sem infectados não há contágio nem recuperação neste passo.
                if (self._congela and not self.vacinacao_pendente()
                        and t - self._t_ultima_dose >= p.espera_parada):
                    # Nada mais muda: repete o último estado até o horizonte.
                    self.n_inf_faixa[t + 1:] = self.n_inf_faixa[t]
                    self.n_naoinf_faixa[t + 1:] = self.n_naoinf_faixa[t]
                    self.imunidade_media[t + 1:] = self.imunidade_media[t]
                    self.n_estado[t + 1:] = self.n_estado[t]
                    self.t_parada = t
                    for s in sorted(set(snapshots) | set(percolacao)):
                        if s > t:
                            ms, mv, mn = self.imunidade(s)
                            registrar(s, ms, self.estados(ms, mv, mn))
                    break
                continue

            self._dinamica(t, m, faixa)

        n_perc = len(percolacao)
        return dict(
            n_inf_faixa=self.n_inf_faixa,
            n_naoinf_faixa=self.n_naoinf_faixa,
            novas_inf_faixa=self.novas_inf_faixa,
            trans_faixa=self.trans_faixa,
            n_estado=self.n_estado,
            doses1=self.doses1,
            reforcos=self.reforcos,
            imunidade_media=self.imunidade_media,
            hist_infeccoes=np.bincount(np.minimum(self.n_infec, MAX_HIST),
                                       minlength=MAX_HIST + 1).astype(np.int32),
            doses1_em_recuperados=np.int32(self.doses1_em_recuperados),
            t_extincao=np.int32(self.t_extincao),
            t_parada=np.int32(self.t_parada),
            grades_tempos=np.array(snapshots, dtype=np.int32),
            grades_m=(np.stack([grades_m[s] for s in snapshots]) if snapshots
                      else np.zeros((0, N, N), dtype=np.float32)),
            grades_infectado=(np.stack([grades_inf[s] for s in snapshots]) if snapshots
                              else np.zeros((0, N, N), dtype=bool)),
            grades_estado=(np.stack([grades_est[s] for s in snapshots]) if snapshots
                           else np.zeros((0, N, N), dtype=np.int8)),
            perc_tempos=np.array(percolacao, dtype=np.int32),
            perc_p=np.array([perc[s]["p"] for s in percolacao], dtype=np.float32),
            perc_pinf=np.array([perc[s]["p_inf"] for s in percolacao], dtype=np.float32),
            perc_n_aglom=np.array([perc[s]["n_aglom"] for s in percolacao], dtype=np.int32),
            perc_envolve=np.array([perc[s]["envolve"] for s in percolacao], dtype=bool),
            perc_s_medio=np.array([perc[s]["s_medio"] for s in percolacao], dtype=np.float32),
            perc_hist=(np.stack([perc[s]["hist"] for s in percolacao]) if n_perc
                       else np.zeros((0, N_CLASSES_AGLOM), dtype=np.int32)),
        )


def simular(p: Parametros, seed: int, snapshots=(), percolacao=(), auditoria: bool = False) -> dict:
    """Roda uma realização e devolve um dicionário de arrays.

    Séries, com ``horizonte + 1`` pontos (o índice é o passo):

    * ``n_inf_faixa``, ``n_naoinf_faixa`` (T+1, 5): infectados e não infectados em cada faixa;
    * ``novas_inf_faixa`` (T+1, 5): infecções ocorridas no passo, pela faixa de quem se infectou;
    * ``trans_faixa`` (T+1, 5): as mesmas infecções, pela faixa de quem transmitiu;
    * ``n_estado`` (T+1, 4): sítios suscetíveis, infectados, recuperados e vacinados;
    * ``doses1``, ``reforcos`` (T+1): doses aplicadas no passo;
    * ``imunidade_media`` (T+1): média de ``m`` na população.

    Totais: ``hist_infeccoes`` (número de indivíduos com 0, 1, ..., MAX_HIST ou
    mais infecções; os infectados iniciais contam), ``doses1_em_recuperados``
    (primeiras doses dadas a quem já tinha se recuperado de uma infecção),
    ``t_extincao`` (primeiro passo sem infectados, -1 se nunca) e ``t_parada``.

    Com ``snapshots``: ``grades_m``, ``grades_infectado`` e ``grades_estado`` nos
    passos de ``grades_tempos``. Com ``percolacao``: ``perc_p``, ``perc_pinf``,
    ``perc_n_aglom``, ``perc_envolve``, ``perc_s_medio`` e ``perc_hist`` nos passos
    de ``perc_tempos`` (veja :func:`medidas_percolacao`).
    """
    return Simulacao(p, seed).rodar(snapshots=snapshots, percolacao=percolacao, auditoria=auditoria)


# ---------------------------------------------------------------------------
# Métricas derivadas (usadas pelos testes e pelas figuras)
# ---------------------------------------------------------------------------
def _dividir(num, den):
    """Divisão que devolve NaN onde o denominador é zero."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    return np.divide(num, den, out=np.full(np.broadcast(num, den).shape, np.nan), where=den > 0)


def mascara_passos(n_inf_faixa: np.ndarray, t_fim=None) -> np.ndarray:
    """Passos que entram nos denominadores de pessoas-passo.

    Um passo ``t`` conta se a dinâmica rodou nele (``t < horizonte``), se havia ao
    menos um infectado no início dele e, quando ``t_fim`` é dado, se ``t < t_fim``.
    Aceita arrays (T+1, 5) ou (n_seeds, T+1, 5).
    """
    I = n_inf_faixa.sum(axis=-1)
    mask = I > 0
    mask[..., -1] = False
    if t_fim is not None:
        mask[..., int(t_fim):] = False
    return mask


def taxas_por_faixa(res: dict, t_fim=None):
    """Taxas por pessoa-passo em cada faixa: ``(lambda_inf, lambda_trans)``.

    ``lambda_inf[b]`` = infecções sofridas na faixa b / pessoas-passo de não infectados na faixa b.
    ``lambda_trans[b]`` = transmissões feitas por infectados da faixa b / pessoas-passo de infectados na faixa b.
    Só contam passos com ao menos um infectado. O resultado é NaN quando o
    denominador da faixa é zero. Funciona para uma execução ou para arrays
    com as seeds no primeiro eixo (uma taxa por seed).
    """
    mask = mascara_passos(res["n_inf_faixa"], t_fim)[..., None]
    lam_inf = _dividir((res["novas_inf_faixa"] * mask).sum(axis=-2),
                       (res["n_naoinf_faixa"] * mask).sum(axis=-2))
    lam_trans = _dividir((res["trans_faixa"] * mask).sum(axis=-2),
                         (res["n_inf_faixa"] * mask).sum(axis=-2))
    return lam_inf, lam_trans


def participacao_vacinados(res: dict, t_fim=None):
    """Fração das infecções sofridas por vacinados e das transmissões feitas por vacinados.

    Devolve ``(frac_infeccoes, frac_transmissoes)``; NaN se não houve infecção
    no intervalo. Com ``t_fim``, só contam os passos ``t < t_fim``.
    """
    fim = None if t_fim is None else int(t_fim)
    inf = res["novas_inf_faixa"][..., :fim, :].sum(axis=-2)
    tr = res["trans_faixa"][..., :fim, :].sum(axis=-2)
    return (_dividir(inf[..., 1:].sum(axis=-1), inf.sum(axis=-1)),
            _dividir(tr[..., 1:].sum(axis=-1), tr.sum(axis=-1)))


def primeira_onda(I_medio: np.ndarray, janela: int = 21, fracao_min: float = 0.10):
    """Delimita a primeira onda a partir da média de I(t) no ensemble.

    Devolve ``(t_pico, t_fim)``: o primeiro máximo local e o primeiro mínimo
    local depois dele. A série é suavizada por uma média móvel centrada de
    ``janela`` passos, e máximos menores que ``fracao_min`` do máximo global
    são ignorados, para que flutuações pequenas não sejam tomadas por uma onda.
    Se não houver mínimo depois do pico (queda monótona até o fim), ``t_fim``
    é o último passo. Se não houver pico algum, devolve ``(0, último passo)``.
    """
    I_medio = np.asarray(I_medio, dtype=float)
    T = I_medio.size - 1
    janela = max(1, min(int(janela), I_medio.size))
    if janela > 1:
        meio = janela // 2
        estendida = np.pad(I_medio, (meio, janela - 1 - meio), mode="edge")
        s = np.convolve(estendida, np.ones(janela) / janela, mode="valid")
    else:
        s = I_medio
    piso = fracao_min * s.max()
    t_pico = None
    for t in range(1, T):
        if s[t] >= piso and s[t] > s[t - 1] and s[t] >= s[t + 1]:
            t_pico = t
            break
    if t_pico is None:
        return 0, T
    for t in range(t_pico + 1, T):
        if s[t] < s[t - 1] and s[t] <= s[t + 1]:
            return t_pico, t
    return t_pico, T
