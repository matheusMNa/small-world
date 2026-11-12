import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.integrate import odeint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import simulate

def run_analysis():
    c_S, c_I, c_R = '#56B4E9', '#D55E00', '#009E73'
    cmap = ListedColormap([c_S, c_I, c_R])
    
    # Agora sim, terminando lindamente no 150
    instantes = [0, 50, 100, 150]
    N = 100
    N_pop = N * N
    steps = 150
    
    # 1. Roda a Simulação Estocástica
    S_hist, I_hist, R_hist, grids = simulate(N=N, init_cluster=3, K_encontros=3, snapshots=instantes, steps=steps)
    
    # =========================================================
    # FIGURA 1: GRADES ESPACIAIS (AS 4 MATRIZES)
    # =========================================================
    fig1, axes = plt.subplots(1, 4, figsize=(18, 5))
    
    for idx, t in enumerate(instantes):
        axes[idx].imshow(grids[t], cmap=cmap, vmin=0, vmax=2)
        axes[idx].set_title(f'Matriz em t = {t}', fontsize=14)
        axes[idx].axis('off')
        
    legend_elements = [Patch(facecolor=c_S, edgecolor='k', label='Suscetível (S)'),
                       Patch(facecolor=c_I, edgecolor='k', label='Infectado (I)'),
                       Patch(facecolor=c_R, edgecolor='k', label='Recuperado (R)')]
    
    fig1.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=14, bbox_to_anchor=(0.5, -0.1))
    fig1.suptitle('Evolução Espacial do Modelo (I0 = 3x3, K = 3)', fontsize=18, y=1.05)
    
    fig1.tight_layout()
    fig1.savefig('analyses/plots/00_analise_geral_grades.png', dpi=300, bbox_inches='tight')
    
    # =========================================================
    # FIGURA 2: GRÁFICO DE LINHAS (ESTOCÁSTICO VS EDO)
    # =========================================================
    alpha = 0.08
    p_rec = 0.05
    W_local = 1.0
    W_mercado = 0.4
    K_encontros = 3
    
    mean_S = 0.6
    mean_T = 0.6
    contatos_efetivos = (8 * W_local) + (K_encontros * W_mercado)
    beta_ode = alpha * mean_S * mean_T * contatos_efetivos
    gamma_ode = p_rec

    def sir_deriv(y, t, N_total, beta, gamma):
        S, I, R = y
        return [-beta * S * I / N_total, 
                 beta * S * I / N_total - gamma * I, 
                 gamma * I]

    # Ajuste: Criar exatos steps + 1 pontos de tempo para bater com o array estocástico
    t_ode = np.linspace(0, steps, steps + 1)
    solution = odeint(sir_deriv, [N_pop - 9, 9, 0], t_ode, args=(N_pop, beta_ode, gamma_ode))
    S_ode, I_ode, R_ode = solution.T
    
    fig2, ax = plt.subplots(figsize=(12, 7))
    
    # Plot EDO (Linhas Tracejadas)
    ax.plot(t_ode, S_ode, '--', color=c_S, alpha=0.7, label='S (EDO SIR Contínuo)')
    ax.plot(t_ode, I_ode, '--', color=c_I, alpha=0.7, label='I (EDO SIR Contínuo)')
    ax.plot(t_ode, R_ode, '--', color=c_R, alpha=0.7, label='R (EDO SIR Contínuo)')
    
    # Plot Estocástico (Linhas Cheias - range também ajustado para steps + 1)
    ax.plot(range(steps + 1), S_hist, '-', color=c_S, linewidth=2.5, label='S (Estocástico Small-World)')
    ax.plot(range(steps + 1), I_hist, '-', color=c_I, linewidth=2.5, label='I (Estocástico Small-World)')
    ax.plot(range(steps + 1), R_hist, '-', color=c_R, linewidth=2.5, label='R (Estocástico Small-World)')
    
    ax.set_title('Comparação da Dinâmica Populacional Entre Modelos', fontsize=16)
    ax.set_xlabel('Tempo (passos)', fontsize=14)
    ax.set_ylabel('População Total', fontsize=14)
    ax.legend(loc='center right', fontsize=11, frameon=False)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    fig2.tight_layout()
    fig2.savefig('analyses/plots/00_analise_geral_linhas.png', dpi=300)
    
    plt.show()

if __name__ == "__main__":
    run_analysis()