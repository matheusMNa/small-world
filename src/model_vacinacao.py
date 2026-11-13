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
* a semente aleatória é um parâmetro e gera três fluxos independentes
  (atributos S0/T0, dinâmica, sorteio de vacinados).

Uso mínimo, a partir da raiz do repositório::

    from src.model_vacinacao import parametros_cenario, simular
    res = simular(parametros_cenario("F1", horizonte=400), seed=0)
    I = res["n_inf_faixa"].sum(axis=1)          # infectados em cada passo
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

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
    espera_parada: int = 30      # passos sem doses antes de permitir a parada antecipada

    def __post_init__(self):
        if self.regime not in ("nenhum", "instantaneo", "continuo"):
            raise ValueError(f"regime desconhecido: {self.regime!r}")
        if self.alocacao not in ("aleatoria", "gaussiana"):
            raise ValueError(f"alocação desconhecida: {self.alocacao!r}")
        if not (self.tau_r < self.tau_d1 and self.tau_r < self.tau_d2):
            raise ValueError("a curva de Bateman exige tau_r < tau_d")
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


def parametros_cenario(codigo: str, **extras) -> Parametros:
    """Parâmetros de um dos 15 cenários (``"F0"`` a ``"F7"``, ``"M0"`` a ``"M6"``).

    ``extras`` sobrescreve qualquer campo, por exemplo ``horizonte=400``.
    """
    if codigo not in CENARIOS:
        raise KeyError(f"cenário desconhecido: {codigo!r}; opções: {', '.join(CENARIOS)}")
    campos = dict(CENARIOS[codigo])
    doenca = campos.pop("doenca")
    base = dict(DOENCAS[doenca])
    base.update(campos)
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

        # Três fluxos independentes derivados da seed: atributos, dinâmica, vacinação.
        # A seed k produz os mesmos S0 e T0 em todos os cenários, e a dinâmica
        # não consome números do fluxo de vacinação (comparações pareadas).
        ss_attr, ss_din, ss_vac = np.random.SeedSequence(self.seed).spawn(3)
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

        # Infectados iniciais: bloco central, como em src/model.py.
        grade = self.infectado.reshape(p.N, p.N)
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
            self._aplicar_dose(self._sortear_primeira_dose(elegiveis, n1), t)
            self._inst_aplicada = True
        elif p.regime == "continuo" and t >= p.t_vac and self._doses1_aplicadas < self._total_continuo:
            elegiveis = np.flatnonzero((self.n_doses == 0) & ~self.infectado)
            n1 = min(p.doses_por_passo, self._total_continuo - self._doses1_aplicadas, elegiveis.size)
            if n1 > 0:
                self._aplicar_dose(self._sortear_primeira_dose(elegiveis, n1), t)
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
    def rodar(self, snapshots=(), auditoria: bool = False) -> dict:
        """Roda até o horizonte e devolve as séries e os totais.

        ``snapshots``: passos em que as grades de ``m`` e de infectados são guardadas.
        ``auditoria``: guarda, para cada reforço, o tempo desde a dose anterior,
        a imunidade e o estado de infecção (usado nos testes).
        """
        p = self.p
        N = p.N
        snapshots = sorted(set(int(s) for s in snapshots if 0 <= int(s) <= p.horizonte))
        grades_m, grades_inf = {}, {}

        for t in range(p.horizonte + 1):
            m, _, _ = self.imunidade(t)
            n1, nref = self._vacinar(t, m, auditoria)
            faixa = self.faixas(m)   # depois da vacinação: quem tomou dose agora já conta como vacinado

            inf = self.infectado
            self.n_inf_faixa[t] = np.bincount(faixa[inf], minlength=N_FAIXAS)
            self.n_naoinf_faixa[t] = np.bincount(faixa[~inf], minlength=N_FAIXAS)
            self.doses1[t] = n1
            self.reforcos[t] = nref
            self.imunidade_media[t] = m.mean()
            n_infectados = int(self.n_inf_faixa[t].sum())
            if n_infectados == 0 and self.t_extincao < 0:
                self.t_extincao = t
            if t in snapshots:
                grades_m[t] = m.reshape(N, N).astype(np.float32)
                grades_inf[t] = inf.reshape(N, N).copy()

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
                    self.t_parada = t
                    for s in snapshots:
                        if s > t:
                            grades_m[s] = self.imunidade(s)[0].reshape(N, N).astype(np.float32)
                            grades_inf[s] = inf.reshape(N, N).copy()
                    break
                continue

            self._dinamica(t, m, faixa)

        return dict(
            n_inf_faixa=self.n_inf_faixa,
            n_naoinf_faixa=self.n_naoinf_faixa,
            novas_inf_faixa=self.novas_inf_faixa,
            trans_faixa=self.trans_faixa,
            doses1=self.doses1,
            reforcos=self.reforcos,
            imunidade_media=self.imunidade_media,
            hist_infeccoes=np.bincount(np.minimum(self.n_infec, MAX_HIST),
                                       minlength=MAX_HIST + 1).astype(np.int32),
            t_extincao=np.int32(self.t_extincao),
            t_parada=np.int32(self.t_parada),
            grades_tempos=np.array(snapshots, dtype=np.int32),
            grades_m=(np.stack([grades_m[s] for s in snapshots]) if snapshots
                      else np.zeros((0, N, N), dtype=np.float32)),
            grades_infectado=(np.stack([grades_inf[s] for s in snapshots]) if snapshots
                              else np.zeros((0, N, N), dtype=bool)),
        )


def simular(p: Parametros, seed: int, snapshots=(), auditoria: bool = False) -> dict:
    """Roda uma realização e devolve um dicionário de arrays.

    Séries, com ``horizonte + 1`` pontos (o índice é o passo):

    * ``n_inf_faixa``, ``n_naoinf_faixa`` (T+1, 5): infectados e não infectados em cada faixa;
    * ``novas_inf_faixa`` (T+1, 5): infecções ocorridas no passo, pela faixa de quem se infectou;
    * ``trans_faixa`` (T+1, 5): as mesmas infecções, pela faixa de quem transmitiu;
    * ``doses1``, ``reforcos`` (T+1): doses aplicadas no passo;
    * ``imunidade_media`` (T+1): média de ``m`` na população.

    Totais: ``hist_infeccoes`` (número de indivíduos com 0, 1, ..., MAX_HIST ou
    mais infecções; os infectados iniciais contam), ``t_extincao`` (primeiro
    passo sem infectados, -1 se nunca) e ``t_parada``.
    """
    return Simulacao(p, seed).rodar(snapshots=snapshots, auditoria=auditoria)


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
