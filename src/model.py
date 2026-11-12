import numpy as np
from scipy.signal import convolve2d

def simulate(N=100, alpha=0.08, p_rec=0.05, steps=150, W_local=1.0, W_mercado=0.4, 
             K_encontros=3, init_cluster=3, init_scattered=0, vac_rate=0.0, vac_time=0, snapshots=None):
    if snapshots is None:
        snapshots = [0, steps]

    np.random.seed(42)
    S_base = np.random.uniform(0.2, 1.0, (N, N))
    T_base = np.random.uniform(0.2, 1.0, (N, N))
    S_curr, T_curr = np.copy(S_base), np.zeros((N, N))

    # --- LÓGICA DE CONDIÇÃO INICIAL (BLOCO VS ESPALHADOS) ---
    if init_scattered > 0:
        # Escolhe posições aleatórias únicas na grade
        idx = np.random.choice(N * N, init_scattered, replace=False)
        row_idx, col_idx = np.unravel_index(idx, (N, N))
        S_curr[row_idx, col_idx] = 0
        T_curr[row_idx, col_idx] = T_base[row_idx, col_idx]
    else:
        # Lógica original de bloco central
        center = N // 2
        r = init_cluster // 2
        if init_cluster == 1:
            S_curr[center, center] = 0
            T_curr[center, center] = T_base[center, center]
        else:
            S_curr[center-r:center+r+1, center-r:center+r+1] = 0
            T_curr[center-r:center+r+1, center-r:center+r+1] = T_base[center-r:center+r+1, center-r:center+r+1]

    if vac_time == 0 and vac_rate > 0:
        vac_mask = (np.random.rand(N, N) < vac_rate) & (S_curr > 0)
        S_curr[vac_mask] = 0 

    S_hist, I_hist, R_hist = [], [], []
    grid_snapshots = {}
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    for t in range(steps + 1):
        if t == vac_time and vac_time > 0 and vac_rate > 0:
            vac_mask = (np.random.rand(N, N) < vac_rate) & (S_curr > 0)
            S_curr[vac_mask] = 0

        S_hist.append(np.sum(S_curr > 0))
        I_hist.append(np.sum(T_curr > 0))
        R_hist.append(np.sum((S_curr == 0) & (T_curr == 0)))

        if t in snapshots:
            grid_state = np.zeros((N, N), dtype=int)
            grid_state[S_curr > 0] = 0
            grid_state[T_curr > 0] = 1
            grid_state[(S_curr == 0) & (T_curr == 0)] = 2
            grid_snapshots[t] = grid_state

        local_viral_pressure = convolve2d(T_curr, kernel, mode='same', boundary='wrap')
        
        if K_encontros > 0:
            T_flat = T_curr.flatten()
            mercado_encounters = np.random.choice(T_flat, size=(N, N, K_encontros))
            mercado_viral_pressure = np.sum(mercado_encounters, axis=2)
        else:
            mercado_viral_pressure = np.zeros((N, N))

        pressao_total = (W_local * local_viral_pressure) + (W_mercado * mercado_viral_pressure)
        prob_infection = 1 - np.exp(-alpha * S_curr * pressao_total)

        rand_inf = np.random.rand(N, N)
        gets_infected = (S_curr > 0) & (rand_inf < prob_infection)

        rand_rec = np.random.rand(N, N)
        recovers = (T_curr > 0) & (rand_rec < p_rec)

        S_curr[gets_infected] = 0
        T_curr[gets_infected] = T_base[gets_infected]
        T_curr[recovers] = 0

    return S_hist, I_hist, R_hist, grid_snapshots