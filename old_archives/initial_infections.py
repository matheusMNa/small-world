import sys
import os
import matplotlib.pyplot as plt

# Garante que a pasta src seja encontrada
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import simulate

def run_analysis():
    plt.figure(figsize=(8,5))
    
    for size in [1, 3, 5, 9]:
        I_hist, _ = simulate(init_cluster=size)
        plt.plot(I_hist, label=f'I0 = {size}x{size} ({size**2} ind.)', linewidth=2)
        
    plt.title('Efeito da Semente Viral Inicial (I0)', fontsize=14)
    plt.xlabel('Tempo (passos)')
    plt.ylabel('Indivíduos Infectados')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig('analyses/plots/01_semente_inicial.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_analysis()