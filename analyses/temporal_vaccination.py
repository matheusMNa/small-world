import sys, os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import simulate

def run_analysis():
    c_S, c_I, c_R = '#56B4E9', '#D55E00', '#009E73'
    cmap = ListedColormap([c_S, c_I, c_R])
    t_final = 150
    t_intervencao = 60 
    
    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(2, 3, height_ratios=[1.3, 1])
    ax_line = fig.add_subplot(gs[0, :])
    
    _, I_base, _, grids_base = simulate(snapshots=[t_final])
    ax_line.plot(I_base, color=c_I, label='Sem Vacina', linewidth=2.5)
    
    _, I_vac_0, _, grids_vac_0 = simulate(vac_rate=0.5, vac_time=0, snapshots=[t_final])
    ax_line.plot(I_vac_0, color='#0072B2', label='Vacinação (50%) em t=0', linewidth=2.5)
    
    _, I_vac_mid, _, grids_vac_mid = simulate(vac_rate=0.5, vac_time=t_intervencao, snapshots=[t_final])
    ax_line.plot(I_vac_mid, color='#CC79A7', label=f'Vacinação (50%) em t={t_intervencao}', linewidth=2.5)
    
    ax_line.axvline(t_intervencao, color='gray', linestyle='--', label=f'Intervenção (t={t_intervencao})')
    ax_line.set_title('Eficácia Temporal: Vacinação Preventiva vs Reativa', fontsize=14)
    ax_line.set_xlabel('Tempo (passos)')
    ax_line.set_ylabel('Indivíduos Infectados')
    ax_line.legend()
    ax_line.grid(True, linestyle=':', alpha=0.7)
    
    cenarios = [("Sem Vacina", grids_base), ("Vacina em t=0", grids_vac_0), (f"Vacina em t={t_intervencao}", grids_vac_mid)]
    for idx, (titulo, grid) in enumerate(cenarios):
        ax_grid = fig.add_subplot(gs[1, idx])
        ax_grid.imshow(grid[t_final], cmap=cmap, vmin=0, vmax=2)
        ax_grid.set_title(f'{titulo}\nEstado Final (t=150)')
        ax_grid.axis('off')
        
    plt.tight_layout()
    plt.savefig('analyses/plots/03_vacinacao_temporal.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_analysis()