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

# 💉 Vacinação com desgaste e reforço 💉

O módulo `src/model_vacinacao.py` estende o modelo acima sem alterar a dinâmica de contágio. Cada indivíduo passa a ter uma imunidade $m \in [0, 1]$, que reduz a suscetibilidade de quem não está infectado e a transmissibilidade de quem está:

$$S_i^{\text{ef}} = S_i^0\,(1 - m_i), \qquad T_i^{\text{ef}} = T_i^0\,(1 - m_i).$$

A imunidade é a maior entre a vacinal e a natural, $m = \max(m_{\text{vac}}, m_{\text{nat}})$. Cada dose gera uma curva de Bateman, com subida rápida e decaimento lento, em que $\tau$ é o número de passos desde a dose:

$$b(\tau) = A\left(e^{-\tau/\tau_d} - e^{-\tau/\tau_r}\right), \qquad \tau^* = \frac{\tau_r\,\tau_d}{\tau_d - \tau_r}\ln\frac{\tau_d}{\tau_r},$$

com $A$ escolhido para que o pico valha 1 em $\tau = \tau^*$. A imunidade vacinal é o máximo das curvas das duas doses mais recentes. A imunidade natural vale 1 no instante da recuperação e decai como $e^{-\tau/\tau_n}$. O modelo é simulado para duas doenças:

| Parâmetro | Flu-like | Measles-like |
|---|---|---|
| $\tau_r$, subida da curva vacinal | 3 | 3 |
| $\tau_d$, primeira dose | 100 | $\infty$ |
| $\tau_d$, doses seguintes | 150 | não se aplica |
| $\tau_n$, imunidade natural | 100 | $\infty$ (permanente) |
| Reforço | sim, com $\tau_b = 20$ | não |

No Flu-like, um vacinado não infectado cuja última dose já passou do pico e cuja imunidade caiu abaixo de 0,5 toma uma dose de reforço, a cada passo, com probabilidade $p_b(u) = 1 - e^{-u/\tau_b}$, em que $u$ é o número de passos desde que $m$ cruzou 0,5. Esses valores são escolhas de modelagem, sem calibração empírica.

A primeira dose segue um de três regimes: 50% dos elegíveis em $t = 0$; 50% dos elegíveis em $t = 60$; ou 50 doses por passo a partir de $t = 0$, até 5000 doses. É elegível quem nunca foi vacinado e não está infectado. A alocação é aleatória ou em grupos de risco, caso em que o sorteio sem reposição tem peso gaussiano na suscetibilidade intrínseca, $w_i \propto \exp[-(S_i^0 - 1)^2 / 2\sigma^2]$ com $\sigma = 0{,}2$. "Grupo de risco" significa apenas maior suscetibilidade à infecção; o modelo não tem gravidade nem mortalidade.

Cada infecção é atribuída a um transmissor por riscos competitivos: dado que $i$ foi infectado, a fonte é o contato $j$ com probabilidade proporcional a $W_{ij}\,T_j^{\text{ef}}$. Infecções e transmissões são contadas por faixa de status no instante do evento: nunca vacinado, ou vacinado com $m \ge 0{,}90$, $0{,}75 \le m < 0{,}90$, $0{,}50 \le m < 0{,}75$ ou $m < 0{,}50$. Como as porcentagens de eventos dependem de quantas pessoas estão em cada faixa, as figuras mostram também as taxas por pessoa-passo, em que só contam os passos com ao menos um infectado.

## Cenários

| Código | Doença | Primeira dose | Alocação | Reforços |
|---|---|---|---|---|
| F0, M0 | Flu-like, Measles-like | sem vacina | n/a | não |
| F1, M1 | Flu-like, Measles-like | $t = 0$ | aleatória | só no Flu-like |
| F2, M2 | Flu-like, Measles-like | $t = 0$ | gaussiana | só no Flu-like |
| F3, M3 | Flu-like, Measles-like | $t = 60$ | aleatória | só no Flu-like |
| F4, M4 | Flu-like, Measles-like | $t = 60$ | gaussiana | só no Flu-like |
| F5, M5 | Flu-like, Measles-like | contínua | aleatória | só no Flu-like |
| F6, M6 | Flu-like, Measles-like | contínua | gaussiana | só no Flu-like |
| F7 | Flu-like | $t = 0$ | aleatória | não (controle) |

O ensemble usa as seeds 0 a 499 em cada cenário e 1500 passos. A seed define os atributos $S^0$ e $T^0$, a dinâmica e o sorteio de vacinados por três fluxos aleatórios independentes, de modo que a seed $k$ produz a mesma população em todos os cenários.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `src/model_vacinacao.py` | Modelo, definição dos 15 cenários e métricas derivadas |
| `analyses/ensemble.py` | Roda os cenários e grava um `.npz` por pedaço de seeds em `results/pedacos/` |
| `analyses/juntar_pedacos.py` | Confere as seeds e reúne os pedaços em `results/<código>.npz` |
| `analyses/figuras_vacinacao.py` | Gera as figuras `05` a `12` e a tabela `resumo_vacinacao.csv` em `analyses/plots/` a partir de `results/`, sem simular |
| `cluster/ensemble_array.slurm` | Job array do SLURM, com campos a preencher (não testado em cluster) |
| `tests/test_model_vacinacao.py` | Testes do modelo |

## Como rodar

Todos os comandos são executados a partir da raiz do repositório, no mesmo ambiente Conda.

1. Testes (cerca de meio minuto):

   ```bash
   python tests/test_model_vacinacao.py
   ```

2. Verificação rápida do fluxo completo, com 3 seeds e 400 passos, em uma pasta temporária:

   ```bash
   python analyses/ensemble.py --seed-fim 3 --horizonte 400 --saida /tmp/teste/pedacos
   python analyses/juntar_pedacos.py --n-seeds 3 --pedacos /tmp/teste/pedacos --saida /tmp/teste
   python analyses/figuras_vacinacao.py --resultados /tmp/teste --saida /tmp/teste/plots --dpi 80
   ```

3. Ensemble completo em uma máquina só, com 8 processos:

   ```bash
   python analyses/ensemble.py --processos 8
   ```

   Ou em um cluster com SLURM, depois de preencher partição, memória, tempo e ambiente em `cluster/ensemble_array.slurm`:

   ```bash
   mkdir -p logs results/pedacos
   sbatch cluster/ensemble_array.slurm
   ```

   Nos dois casos, pedaços já gravados são pulados, então basta relançar o mesmo comando para retomar uma execução interrompida.

4. Junção e figuras:

   ```bash
   python analyses/juntar_pedacos.py
   python analyses/figuras_vacinacao.py
   ```

   A junção falha, dizendo quais seeds faltam ou se repetem, se algum cenário não tiver exatamente as seeds 0 a 499. Cada figura com banda de erro sai em duas versões: `_dp` (desvio padrão entre seeds) e `_ep` (erro padrão da média).

Para usar o modelo diretamente:

```python
from src.model_vacinacao import parametros_cenario, simular

res = simular(parametros_cenario("F1", horizonte=400), seed=0)
infectados = res["n_inf_faixa"].sum(axis=1)
```

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
