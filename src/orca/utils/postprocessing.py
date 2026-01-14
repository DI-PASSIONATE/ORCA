import skrf as rf
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

def s_param_dict_to_network(s_param_dict: dict, frequencies: np.ndarray) -> tuple[int, rf.Network, dict]:
    N = int(np.sqrt(len(s_param_dict) // 2))  # number of ports

    nb_f = len(frequencies)

    # Check frequency length
    if nb_f < 1:
        raise ValueError("Frequency array must have at least one element.")

    # Initialize S-matrix of shape (nb_f, N, N)
    S = np.zeros((nb_f, N, N), dtype=np.complex64)

    # Fill S-matrix
    for i in range(N):
        for j in range(N):
            real = np.array(s_param_dict[f"S{i+1}{j+1}_real"], dtype=np.float32)
            imag = np.array(s_param_dict[f"S{i+1}{j+1}_imag"], dtype=np.float32)
            if real.shape[0] != nb_f or imag.shape[0] != nb_f:
                raise ValueError(f"S{i+1}{j+1} length mismatch with frequency array.")
            S[:, i, j] = real + 1j * imag  # note: frequency as first dimension

    # Create skrf Network object
    ntwk = rf.Network(frequency=frequencies, s=S, f_unit='GHz')
    ntwk.frequency.unit = 'GHz'

    merged_output = {}
    for i in range(N):
        for j in range(N):
            merged_output[f"S{i+1}{j+1}"] = S[:, i, j]

    return N, ntwk, merged_output

def single_ended_to_mixed_mode(ntwk: rf.Network) -> rf.Network:
    """
    Converts a 4-port single-ended network to a 2-port mixed-mode network using rf.se2gmm.
    Usually port 1 and 2 are considered differential pair 1, and port 3 and 4 differential pair 2.
    Args:
        network (rf.Network): 4-port single-ended network.
    Returns:
        rf.Network: 2-port mixed-mode network.
    """
    ntwk.se2gmm(p=2)
    return ntwk.nports, ntwk

def plot_diff_s_params_and_k(ntwk: rf.Network):
    """
    Plots the differential S-parameters and coupling factor k for a 4-port single-ended network.
    Args:
        network (rf.Network): 4-port single-ended network.
    """
    # Calculate k
    z = ntwk.z
    ImZ12 = np.imag(z[:, 0, 1])
    ImZ11 = np.imag(z[:, 0, 0])
    ImZ22 = np.imag(z[:, 1, 1])

    with np.errstate(divide='ignore', invalid='ignore'):
        # We use ntwk.f/1e9 to explicitly ensure the X-axis is in GHz
        freq_ghz = ntwk.f
        k_vs_f = np.abs(ImZ12) / np.sqrt(np.abs(ImZ11 * ImZ22))

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Primary Y-Axis (S-parameters)
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('S-Parameters (dB)')
    ntwk.plot_s_db(m=1, n=0, ax=ax1, label='Insertion Loss ($S_{d2d1}$)')
    ntwk.plot_s_db(m=0, n=0, ax=ax1, label='Return Loss ($S_{d1d1}$)')
    ntwk.plot_s_db(m=3, n=0, ax=ax1, label='Mode Conversion ($S_{c2d1}$)')

    

    # Secondary Y-Axis (k)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Coupling Factor ($k$)', color='red')
    # Use freq_ghz here instead of f_scaled
    ax2.plot(freq_ghz, k_vs_f, color='red', linewidth=2, label='Coupling Factor ($k$)')
    ax2.set_ylim(0, 1.1) 
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('Octagon Transformer: Mixed-Mode S-Params & Coupling')
    fig.tight_layout()
    plt.show()