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
    
    # Os 5 cenários combinando a progressão de blocos e a semente espalhada
    cenarios = [
        ("I0 = 1x1 (1 ind.)", {'init_cluster': 1, 'init_scattered': 0}),
        ("I0 = 3x3 (9 ind.)", {'init_cluster': 3, 'init_scattered': 0}),
        ("I0 = 5x5 (25 ind.)", {'init_cluster': 5, 'init_scattered': 0}),
        ("I0 = 9x9 (81 ind.)", {'init_cluster': 9, 'init_scattered': 0}),
        ("I0 = 81 (espalhados)", {'init_cluster': 0, 'init_scattered': 81})
    ]
    
    # Aumentei a largura da figura para acomodar as 5 matrizes lado a lado
    fig = plt.figure(figsize=(18, 9))
    gs = GridSpec(2, 5, height_ratios=[1.3, 1])
    
    ax_line = fig.add_subplot(gs[0, :])
    
    for idx, (label, kwargs) in enumerate(cenarios):
        # Roda a simulação para o cenário atual
        _, I_hist, _, grids = simulate(**kwargs, snapshots=[t_snapshot])
        
        # Plota a curva com destaque para os espalhados
        estilo = '--' if 'espalhados' in label else '-'
        espessura = 3 if 'espalhados' in label else 2.5
        ax_line.plot(I_hist, linestyle=estilo, linewidth=espessura, label=label)
        
        # Plota a matriz correspondente na parte inferior
        ax_grid = fig.add_subplot(gs[1, idx])
        ax_grid.imshow(grids[t_snapshot], cmap=cmap, vmin=0, vmax=2)
        ax_grid.set_title(f'{label}\nt={t_snapshot}', fontsize=11)
        ax_grid.axis('off')
        
    ax_line.set_title('Efeito da Semente Viral Inicial (I0)', fontsize=15)
    ax_line.set_xlabel('Tempo (Passos)', fontsize=12)
    ax_line.set_ylabel('Indivíduos Infectados (I)', fontsize=12)
    ax_line.legend(loc='upper right', fontsize=11)
    ax_line.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('analyses/plots/01_semente_inicial.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_analysis()