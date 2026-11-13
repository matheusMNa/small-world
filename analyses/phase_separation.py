import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import simulate

def run_analysis():
    c_S, c_I, c_R = '#56B4E9', '#D55E00', '#009E73'
    c_cinza = '#E0E0E0' 
    
    cmap_normal = ListedColormap([c_S, c_I, c_R])
    
    # IGNORAMOS AS LINHAS COMPLETAMENTE AQUI (_, _, _, grids_v)
    _, _, _, grids_v = simulate(vac_rate=0.5, vac_time=0, snapshots=[0, 150])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(grids_v[0], cmap=cmap_normal, vmin=0, vmax=2)
    axes[0].set_title('Microestado Inicial (t=0)', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(grids_v[150], cmap=cmap_normal, vmin=0, vmax=2)
    axes[1].set_title('Estado Final (t=150)', fontsize=14)
    axes[1].axis('off')
    
    grid_filtrada = np.copy(grids_v[150])
    filtro_visual = np.where(grid_filtrada == 0, 1, 0)
    cmap_filtro = ListedColormap([c_cinza, c_S])
    
    axes[2].imshow(filtro_visual, cmap=cmap_filtro, vmin=0, vmax=1)
    axes[2].set_title('Separação de Fase Evidente', fontsize=14)
    axes[2].axis('off')
    
    plt.suptitle('Teoria da Percolação', fontsize=16, y=1.05)
    
    legend_elements = [
        Patch(facecolor=c_S, edgecolor='k', label='Suscetíveis Preservados (Clusters)'),
        Patch(facecolor=c_I, edgecolor='k', label='Infectados Ativos'),
        Patch(facecolor=c_R, edgecolor='k', label='Recuperados / Vacinados'),
        Patch(facecolor=c_cinza, edgecolor='k', label='Material Inerte (Vac+Rec)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=12, bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    plt.savefig('analyses/plots/04_separacao_fase.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    run_analysis()