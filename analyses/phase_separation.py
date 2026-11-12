import sys, os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import simulate

def run_analysis():
    c_S, c_I, c_R = '#56B4E9', '#D55E00', '#009E73'
    cmap = ListedColormap([c_S, c_I, c_R])
    
    # Pedindo a matriz 150
    _, _, _, grids_v = simulate(vac_rate=0.4, vac_time=0, snapshots=[0, 150]) 
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].imshow(grids_v[0], cmap=cmap, vmin=0, vmax=2)
    axes[0].set_title('Início (t=0)', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(grids_v[150], cmap=cmap, vmin=0, vmax=2)
    axes[1].set_title('Fim da Epidemia (t=150)', fontsize=14)
    axes[1].axis('off')
    
    plt.suptitle('Separação de Fase e Imunidade de Rebanho Estrutural', fontsize=16, y=1.05)
    
    legend_elements = [
        Patch(facecolor=c_S, edgecolor='k', label='Suscetíveis Ativos (Azul)'),
        Patch(facecolor=c_I, edgecolor='k', label='Infectados (Laranja)'),
        Patch(facecolor=c_R, edgecolor='k', label='Recuperados / Vacinados (Verde)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12, bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    plt.savefig('analyses/plots/04_separacao_fase.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    run_analysis()