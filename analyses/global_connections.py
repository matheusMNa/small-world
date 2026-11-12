import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import simulate

def run_analysis():
    plt.figure(figsize=(8,5))
    
    for K in [0, 1, 3, 10]:
        I_hist, _ = simulate(K_encontros=K)
        plt.plot(I_hist, label=f'K = {K} encontros casuais/passo', linewidth=2)
        
    plt.title('Influência de Conexões Globais', fontsize=14)
    plt.xlabel('Tempo (passos)')
    plt.ylabel('Indivíduos Infectados')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig('analyses/plots/02_conexoes_globais.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_analysis()