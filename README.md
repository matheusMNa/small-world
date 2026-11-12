# Small-World: Propagação de Doenças em Autômato Celular

Este repositório implementa um modelo estocástico de propagação de doenças em uma grade $N \times N$ de indivíduos, tratado como um autômato celular com ingredientes da termodinâmica estatística. A pergunta abordada é como a combinação de contatos próximos (vizinhança de Moore) com encontros esporádicos aleatórios, em analogia a interações como as de um mercado, altera a dinâmica de uma epidemia em relação a uma rede puramente local e ao campo médio de um modelo SIR em que todos interagem com todos. A abordagem é uma simulação estocástica do tipo Monte Carlo, na qual cada indivíduo carrega suscetibilidade e transmissibilidade intrínsecas e os contatos seguem uma estrutura inspirada em redes *small-world*. O código do modelo está em `src/model.py`, e o diretório `analyses/` reúne os scripts de análise (semente inicial, conexões globais, vacinação e comparação com o SIR em campo médio) e as figuras geradas em `analyses/plots/`.

![Badge Ilum](https://img.shields.io/badge/Ilum%20-%20purple) ![Badge Status](https://img.shields.io/badge/Status%20-%20Em_Desenvolvimento%20-%20orange)

---

# 📐 O modelo 📐

Cada sítio da grade é um indivíduo descrito por uma tupla $(S, T)$, em que $S$ é a suscetibilidade e $T$ é a transmissibilidade. Os valores intrínsecos $S_i^0$ e $T_i^0$ de cada indivíduo são sorteados uma única vez de uma distribuição uniforme em $[0{,}2;\,1{,}0]$ e permanecem fixos durante a simulação (desordem congelada). Há três tipos de indivíduos:

| Estado | Tupla | Significado |
|---|---|---|
| Suscetível | $(S, 0)$ | Pode contrair a doença, mas ainda não a possui e, portanto, não a transmite. |
| Infectado | $(0, T)$ | Transmite a doença e não pode contraí-la novamente. |
| Recuperado (ou vacinado) | $(0, 0)$ | Não transmite nem contrai a doença, pelas hipóteses do modelo. |

A cada passo de tempo, a atualização é síncrona. A pressão viral sentida pelo indivíduo $i$ combina um termo local, dado pelos 8 vizinhos de Moore (com contornos periódicos), e um termo de encontros casuais:

$$P_i(t) = W_L \sum_{j \in \mathcal{V}_i} T_j(t) + W_M \sum_{k=1}^{K} T_{j_k}(t),$$

em que $\mathcal{V}_i$ é a vizinhança de Moore de $i$ e os $K$ indivíduos $j_k$ são sorteados uniformemente em toda a população, de novo a cada passo, com peso $W_M < W_L$ por representarem contatos mais breves. Um suscetível é infectado com probabilidade

$$p_i^{\text{inf}} = 1 - \exp\!\left(-\alpha\, S_i\, P_i\right),$$

passando de $(S_i^0, 0)$ para $(0, T_i^0)$. Um infectado se recupera com probabilidade $p_{\text{rec}}$ a cada passo, passando de $(0, T_i^0)$ para $(0, 0)$.

Diferentemente do modelo de Watts-Strogatz, em que os atalhos aleatórios são fixos, os encontros casuais aqui são sorteados novamente a cada passo (topologia *annealed*). Os parâmetros padrão de `simulate` são $N = 100$, $\alpha = 0{,}08$, $p_{\text{rec}} = 0{,}05$, $W_L = 1{,}0$, $W_M = 0{,}4$, $K = 3$, 150 passos, semente aleatória 42 e uma semente inicial de infectados $3 \times 3$ no centro da grade. A vacinação, quando ativada, leva uma fração `vac_rate` dos suscetíveis ao estado $(0, 0)$ no instante `vac_time`.

Para comparação, `analyses/global_analysis.py` resolve um SIR de campo médio com $\beta = \alpha\,\bar{S}\,\bar{T}\,(8 W_L + K W_M)$ e $\gamma = p_{\text{rec}}$, usando $\bar{S} = \bar{T} = 0{,}6$, isto é, a média da distribuição uniforme dos atributos.

## Estrutura do repositório

| Arquivo | Conteúdo | Figura gerada |
|---|---|---|
| `src/model.py` | Função `simulate`, que retorna os históricos $S(t)$, $I(t)$, $R(t)$ e as grades em instantes escolhidos | n/a |
| `analyses/global_analysis.py` | Evolução espacial e comparação entre a simulação estocástica e o SIR em campo médio | `00_analise_geral_grades.png`, `00_analise_geral_linhas.png` |
| `analyses/initial_infections.py` | Efeito do tamanho da semente inicial de infectados | `01_semente_inicial.png` |
| `analyses/global_connections.py` | Efeito do número $K$ de encontros casuais por passo | `02_conexoes_globais.png` |
| `analyses/temporal_vaccination.py` | Vacinação preventiva ($t = 0$) versus reativa ($t = 60$) | `03_vacinacao_temporal.png` |
| `analyses/phase_separation.py` | Separação de fase e imunidade de rebanho estrutural com 40% de vacinação inicial | `04_separacao_fase.png` |
| `old_archives/thermodynamic_model.py` | Versão anterior do modelo, em um único script | n/a |

---

# 🛠️ Tecnologias utilizadas 🛠️

## IA

O Google Gemini Pro 3.1 foi utilizado no desenvolvimento dos scripts do projeto.

## Bibliotecas e módulos

As principais bibliotecas, módulos e pacotes utilizados foram:

- [Numpy](https://numpy.org/)
- [SciPy](https://docs.scipy.org/doc/scipy/) (`scipy.signal.convolve2d` e `scipy.integrate.odeint`)
- [Matplotlib](https://matplotlib.org/)
- [Conda](https://docs.conda.io/en/latest/)

**Ambiente virtual.** O projeto foi desenvolvido com Python 3.12 em um ambiente virtual Conda. O repositório ainda não possui um arquivo `environment.yml`, de modo que o ambiente é criado diretamente pelo comando indicado na seção seguinte.

---

# 💻 Instalação e Instruções 💻

Pré-requisito: [Conda](https://docs.conda.io/en/latest/) instalado. Para executar o projeto:

1. Clone o repositório e acesse a pasta:

   ```bash
   git clone https://github.com/matheusMNa/small-world.git
   cd small-world
   ```

2. Crie o ambiente virtual com as dependências:

   ```bash
   conda create -n small-world python=3.12 numpy scipy matplotlib
   ```

3. Ative o ambiente:

   ```bash
   conda activate small-world
   ```

4. A partir da raiz do repositório, execute o script de análise desejado, por exemplo:

   ```bash
   python analyses/global_analysis.py
   ```

   Os scripts devem ser executados da raiz, pois importam `src.model` e salvam as figuras no caminho relativo `analyses/plots/`. Cada script também chama `plt.show()`; em um ambiente sem interface gráfica, defina `MPLBACKEND=Agg` antes do comando para que as figuras sejam apenas salvas.

Para usar o modelo diretamente em outro script ou terminal, também a partir da raiz:

```python
from src.model import simulate

S_hist, I_hist, R_hist, grids = simulate(N=100, K_encontros=3, vac_rate=0.0, snapshots=[0, 75, 150])
```

---

# 👥 Desenvolvedor do Projeto 👥

[<img loading="lazy" src="https://github.com/Caio1833.png" width=115><br><sub>Caio Ávila Paulo</sub>](https://github.com/Caio1833)

Quarto semestre, na Ilum Escola de Ciência

[<img loading="lazy" src="https://github.com/giovaninagano5.png" width=115><br><sub>Giovani Massayuki Miranda Nagano</sub>](https://github.com/giovaninagano5)

Quarto semestre, na Ilum Escola de Ciência

[<img loading="lazy" src="https://github.com/matheusMNa.png" width=115><br><sub>Matheus Macedo do Nascimento</sub>](https://github.com/matheusMNa)

Quarto semestre, na Ilum Escola de Ciência

---

# 🙏 Agradecimentos 🙏

Agradecimento ao professor da matéria de Fundamentos Estatísticos da Termodinâmica:

📍Professor Felipe David Crasto de Lima
