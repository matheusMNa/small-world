import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from scipy.signal import convolve2d
from scipy.integrate import odeint

# --- Parâmetros da Simulação ---
N = 100
N_pop = N * N
alpha = 0.08          # Constante de acoplamento de infecção reduzida
p_rec = 0.05          # Probabilidade de recuperação
steps = 150
snapshots = [0, 50, 100, 150]

# --- Pesos de Interação (A Inovação do seu Modelo) ---
W_local = 1.0         # Peso forte para a vizinhança diária
W_mercado = 0.4       # Peso menor para encontros casuais (tempo de contato menor)
K_encontros = 3       # Número de pessoas aleatórias que cada um encontra por passo

np.random.seed(42)

# Desordem Congelada (Suscetibilidade e Transmissibilidade intrínsecas)
S_base = np.random.uniform(0.2, 1.0, (N, N))
T_base = np.random.uniform(0.2, 1.0, (N, N))

S_curr = np.copy(S_base)
T_curr = np.zeros((N, N))

# Condição inicial: Cluster central infectado
center = N // 2
S_curr[center-1:center+2, center-1:center+2] = 0
T_curr[center-1:center+2, center-1:center+2] = T_base[center-1:center+2, center-1:center+2]

S_hist, I_hist, R_hist = [], [], []
saved_grids = {}

kernel = np.array([[1, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]])

# --- 1. Dinâmica Estocástica de Contato Misto ---
for t in range(steps + 1):
    if t < steps:
        S_hist.append(np.sum(S_curr > 0))
        I_hist.append(np.sum(T_curr > 0))
        R_hist.append(np.sum((S_curr == 0) & (T_curr == 0)))
        
    if t in snapshots:
        grid_state = np.zeros((N, N), dtype=int)
        grid_state[S_curr > 0] = 0
        grid_state[T_curr > 0] = 1
        grid_state[(S_curr == 0) & (T_curr == 0)] = 2
        saved_grids[t] = grid_state
        
    if t == steps: break

    # COMPONENTE 1: Família/Trabalho (Topologia Fixa)
    local_viral_pressure = convolve2d(T_curr, kernel, mode='same', boundary='wrap')
    
    # COMPONENTE 2: Mercado/Transporte (Topologia Flutuante Annealed)
    # Sorteia K_encontros aleatórios para CADA indivíduo da grade simultaneamente
    T_flat = T_curr.flatten()
    mercado_encounters = np.random.choice(T_flat, size=(N, N, K_encontros))
    mercado_viral_pressure = np.sum(mercado_encounters, axis=2)
    
    # Probabilidade Final de Infecção combinando os dois mundos
    pressao_total = (W_local * local_viral_pressure) + (W_mercado * mercado_viral_pressure)
    prob_infection = 1 - np.exp(-alpha * S_curr * pressao_total)
    
    # Transições
    rand_inf = np.random.rand(N, N)
    gets_infected = (S_curr > 0) & (rand_inf < prob_infection)
    
    rand_rec = np.random.rand(N, N)
    recovers = (T_curr > 0) & (rand_rec < p_rec)
    
    S_curr[gets_infected] = 0
    T_curr[gets_infected] = T_base[gets_infected]
    T_curr[recovers] = 0

# --- 2. Solução Numérica do Modelo EDO (Campo Médio) ---
# A taxa de contato efetiva global da EDO precisa refletir ambas as pressões
contatos_efetivos = (8 * W_local) + (K_encontros * W_mercado)
beta_ode = alpha * np.mean(S_base) * np.mean(T_base) * contatos_efetivos
gamma_ode = p_rec

def sir_deriv(y, t, N_total, beta, gamma):
    S, I, R = y
    return [-beta * S * I / N_total, 
             beta * S * I / N_total - gamma * I, 
             gamma * I]

solution = odeint(sir_deriv, [N_pop - 9, 9, 0], np.linspace(0, steps, steps), args=(N_pop, beta_ode, gamma_ode))
S_ode, I_ode, R_ode = solution.T

# --- 3. Visualização ---
c_S, c_I, c_R = '#56B4E9', '#D55E00', '#009E73'
cmap = ListedColormap([c_S, c_I, c_R])

fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 4, height_ratios=[1, 1.3], figure=fig)

for idx, t in enumerate(snapshots):
    ax = fig.add_subplot(gs[0, idx])
    ax.imshow(saved_grids[t], cmap=cmap, vmin=0, vmax=2)
    ax.set_title(f'Matriz em t = {t}', fontsize=12)
    ax.axis('off')

ax_lines = fig.add_subplot(gs[1, :])
ax_lines.plot(np.linspace(0, steps, steps), S_ode, '--', color=c_S, alpha=0.6, label='S (EDO: Macroscópico)')
ax_lines.plot(np.linspace(0, steps, steps), I_ode, '--', color=c_I, alpha=0.6, label='I (EDO: Macroscópico)')
ax_lines.plot(np.linspace(0, steps, steps), R_ode, '--', color=c_R, alpha=0.6, label='R (EDO: Macroscópico)')

ax_lines.plot(range(steps), S_hist, '-', color=c_S, linewidth=2.5, label='S (Misto: Vizinhança + Mercado)')
ax_lines.plot(range(steps), I_hist, '-', color=c_I, linewidth=2.5, label='I (Misto: Vizinhança + Mercado)')
ax_lines.plot(range(steps), R_hist, '-', color=c_R, linewidth=2.5, label='R (Misto: Vizinhança + Mercado)')

ax_lines.set_title('Dinâmica Small-World: Heterogeneidade Intrínseca com Contatos Locais e Casuais', fontsize=14)
ax_lines.set_xlabel('Tempo (Passos)', fontsize=12)
ax_lines.set_ylabel('População', fontsize=12)
ax_lines.legend(loc='upper right', frameon=False, fontsize=10, ncol=2)
ax_lines.grid(True, linestyle=':', alpha=0.7)

legend_elements = [Patch(facecolor=c_S, edgecolor='black', label='Suscetível (S)'),
                   Patch(facecolor=c_I, edgecolor='black', label='Infectado (I)'),
                   Patch(facecolor=c_R, edgecolor='black', label='Recuperado (R)')]
fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=11, bbox_to_anchor=(0.5, 0.98))

plt.suptitle('Epidemia em Rede de Mundo Pequeno (Small-World)', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()