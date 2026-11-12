import sys, os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import simulate

def run_analysis():
    c_S, c_I, c_R = '#56B4E9', '#D55E00', '#009E73'
    cmap = ListedColormap([c_S, c_I, c_R])
    t_snapshot = 60 
    ks = [0, 1, 3, 10]
    
    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(2, 4, height_ratios=[1.3, 1])
    
    ax_line = fig.add_subplot(gs[0, :])
    for idx, K in enumerate(ks):
        # AQUI ESTÁ A CORREÇÃO: _, I_hist, _, grids
        _, I_hist, _, grids = simulate(K_encontros=K, snapshots=[t_snapshot])
        ax_line.plot(I_hist, linewidth=2.5, label=f'K = {K} encontros casuais/passo')
        
        ax_grid = fig.add_subplot(gs[1, idx])
        ax_grid.imshow(grids[t_snapshot], cmap=cmap, vmin=0, vmax=2)
        ax_grid.set_title(f'Grade para K={K} (t={t_snapshot})')
        ax_grid.axis('off')
        
    ax_line.set_title('Influência de Conexões Globais', fontsize=14)
    ax_line.set_xlabel('Tempo (passos)\n \n')
    ax_line.set_ylabel('Indivíduos Infectados')
    ax_line.legend()
    ax_line.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('analyses/plots/02_conexoes_globais.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_analysis()