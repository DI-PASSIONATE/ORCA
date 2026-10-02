import matplotlib.pyplot as plt
import numpy as np
import skrf as rf

from orca.logger import logger


def to_mixed_mode(ntwk):
    """Mixed-mode view of a single-ended network, as a copy.

    Returned separately rather than alongside the electrical parameters: callers
    iterate over that dict computing per-curve errors, and a Network is not a
    curve.
    """
    mm_ntwk = ntwk.copy()
    if ntwk.nports >= 4:
        mm_ntwk.se2gmm(p=ntwk.nports // 2)
    return mm_ntwk


def calculate_electrical_parameters(ntwk):
    # 1. Single-ended to Mixed-Mode Conversion
    mm_ntwk = to_mixed_mode(ntwk)

    freq_ghz = ntwk.f / 1e9
    omega = 2 * np.pi * ntwk.f

    # Extract Differential Z-parameters for lumped metrics
    # Index 0 = Primary Diff (d1), Index 1 = Secondary Diff (d2)
    z_d11 = mm_ntwk.z[:, 0, 0]
    z_d22 = mm_ntwk.z[:, 1, 1]
    z_d12 = mm_ntwk.z[:, 0, 1]

    # Calculate Parameters
    with np.errstate(divide="ignore", invalid="ignore"):
        Lp = np.imag(z_d11) / omega * 1e9
        Ls = np.imag(z_d22) / omega * 1e9
        Rp, Rs = np.real(z_d11), np.real(z_d22)
        Qp = np.imag(z_d11) / np.real(z_d11)
        Qs = np.imag(z_d22) / np.real(z_d22)
        k = np.abs(np.imag(z_d12)) / np.sqrt(np.abs(np.imag(z_d11) * np.imag(z_d22)))

    im = np.imag(z_d11)
    cross = np.where(im[:-1] * im[1:] < 0)[0]   # actual sign changes only

    f_min = 20.0
    cross = cross[freq_ghz[cross] >= f_min]

    if cross.size == 0:
        # NaN rather than None, so callers can do arithmetic on the result and
        # filter it out with the usual isfinite() masks.
        srf_f = float("nan")
    else:
        # Note: this index must not be called k - that name holds the coupling
        # coefficient computed above.
        cross_idx = cross[0]
        f0, f1 = freq_ghz[cross_idx], freq_ghz[cross_idx + 1]
        y0, y1 = im[cross_idx], im[cross_idx + 1]
        srf_f = float(f0 - y0 * (f1 - f0) / (y1 - y0))

    return {
        "Lp": np.array(Lp),
        "Ls": np.array(Ls),
        "Rp": np.array(Rp),
        "Rs": np.array(Rs),
        "Qp": np.array(Qp),
        "Qs": np.array(Qs),
        "k": np.array(k),
        "z_d11": np.array(z_d11),
        "srf_f": np.array(srf_f),
    }


def pointwise_relative_error(pred, gt) -> np.ndarray:
    """
    Relative error in percent at every point of a predicted and a reference curve.

    The error is taken against the local magnitude, floored at 1% of the curve's median
    magnitude so zero crossings stay finite. See :func:`median_relative_error` for why.

    Args:
        pred: Predicted values (array or scalar).
        gt: Reference values (array or scalar), same shape as pred.

    Returns:
        np.ndarray: Error per point in percent; NaN where it is not finite.
    """
    pred = np.atleast_1d(pred)
    gt = np.atleast_1d(gt)

    gt_abs = np.abs(gt)
    finite_gt = gt_abs[np.isfinite(gt_abs)]
    scale = np.median(finite_gt) if finite_gt.size else 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        errors = np.abs(pred - gt) / np.maximum(gt_abs, 0.01 * scale + 1e-10) * 100

    return np.where(np.isfinite(errors), errors, np.nan)


def median_relative_error(pred, gt) -> float:
    """
    Median relative error, in percent, between a predicted and a reference curve.

    Normalizing by the mean of the whole curve hides real error: quantities like L and Q
    diverge at self-resonance, and those few huge values dominate the mean, driving the
    reported error towards zero. Instead the error is taken per point against the local
    magnitude - floored at 1% of the curve's median magnitude so zero crossings stay
    finite - and aggregated with a median, which the divergent points cannot dominate.

    Args:
        pred: Predicted values (array or scalar).
        gt: Reference values (array or scalar), same shape as pred.

    Returns:
        float: Median relative error in percent, or NaN if nothing finite remains.
    """
    errors = pointwise_relative_error(pred, gt)
    errors = errors[np.isfinite(errors)]
    return float(np.median(errors)) if errors.size else float("nan")


def plot_rfic_transformer_metrics(ntwk):
    metrics = calculate_electrical_parameters(ntwk)
    mm_ntwk = to_mixed_mode(ntwk)
    freq = ntwk.f / 1e9
    Lp, Ls = metrics["Lp"], metrics["Ls"]
    Rp, Rs = metrics["Rp"], metrics["Rs"]
    Qp, Qs = metrics["Qp"], metrics["Qs"]
    k = metrics["k"]
    z_d11 = metrics["z_d11"]

    # Setup Plot
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(
        f"RFIC Transformer Report: {ntwk.name}", fontsize=16, fontweight="bold"
    )

    # Subplot 1: S-Parameters (S11 & S21 Mixed-Mode)
    axes[0, 0].plot(
        freq,
        mm_ntwk.s_db[:, 1, 0],
        label="Sdd21 (Insertion Loss)",
        color="teal",
        lw=2.5,
    )
    axes[0, 0].plot(
        freq,
        mm_ntwk.s_db[:, 0, 0],
        label="Sdd11 (Return Loss)",
        color="darkorange",
        ls="--",
    )
    axes[0, 0].set_title("Mixed-Mode S-Parameters", fontsize=14)
    axes[0, 0].set_ylabel("Magnitude [dB]")
    axes[0, 0].legend()

    # Subplot 2: Inductance (Lp & Ls)
    axes[0, 1].plot(freq, Lp, label="Lp (Primary)", color="blue")
    axes[0, 1].plot(freq, Ls, label="Ls (Secondary)", color="cyan")
    axes[0, 1].set_title("Inductance [nH]", fontsize=14)
    axes[0, 1].set_ylabel("L [nH]")
    axes[0, 1].legend()

    # Subplot 3: Quality Factor (Qp & Qs)
    axes[1, 0].plot(freq, Qp, label="Qp (Primary)", color="red")
    axes[1, 0].plot(freq, Qs, label="Qs (Secondary)", color="magenta")
    axes[1, 0].set_title("Quality Factor (Q)", fontsize=14)
    axes[1, 0].set_ylabel("Q")
    axes[1, 0].legend()

    # Subplot 4: Resistance (Rp & Rs)
    axes[1, 1].plot(freq, Rp, label="Rp (Primary)", color="darkgreen")
    axes[1, 1].plot(freq, Rs, label="Rs (Secondary)", color="lime")
    axes[1, 1].set_title("Loss / Resistance [Ω]", fontsize=14)
    axes[1, 1].set_ylabel("R [Ω]")
    axes[1, 1].legend()

    # Subplot 5: Coupling Coefficient (k)
    axes[2, 0].plot(freq, k, color="purple", lw=2)
    axes[2, 0].set_title("Coupling Coefficient (k)", fontsize=14)
    axes[2, 0].set_ylabel("k")
    axes[2, 0].set_ylim(0, 1.1)

    # Subplot 6: Reactance & SRF Identification
    axes[2, 1].plot(freq, np.imag(z_d11), label="Im(Zdd11)", color="brown")
    axes[2, 1].axhline(0, color="black", lw=1)  # y=0 line to find zero-crossing
    srf_f = metrics["srf_f"]
    if np.isfinite(srf_f):
        axes[2, 1].axvline(
            srf_f, color="red", linestyle=":", label=f"SRF: {srf_f:.2f} GHz"
        )
    axes[2, 1].set_title("Primary Reactance & SRF", fontsize=14)
    axes[2, 1].set_ylabel("Im(Z) [Ω]")
    axes[2, 1].legend()

    for ax in axes.flat:
        ax.set_xlabel("Frequency [GHz]")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def _infer_port_count(s_param_dict: dict) -> int:
    """
    Number of ports described by a dict of ``S<i><j>_real``/``_imag`` keys.

    Read off the highest port index appearing in the keys rather than from the
    number of entries, so that a full N x N output and an upper-triangle-only one
    are told apart (both counts collide for some N: 36 entries is either an
    8-port triangle or a 6-port full matrix).
    """
    ports = [
        max(int(key[1]), int(key[2]))
        for key in s_param_dict
        if len(key) > 2 and key[0] == "S" and key[1:3].isdigit()
    ]
    if not ports:
        raise ValueError(
            "No S-parameter entries found: expected keys such as 'S11_real'. "
            "Note that this naming only distinguishes up to 9 ports."
        )
    return max(ports)


def s_param_dict_to_network(
    s_param_dict: dict, frequencies: np.ndarray
) -> tuple[int, rf.Network, dict]:
    """
    Build a network from a dict of named real/imaginary columns, as an ONNX model emits.

    Both output layouts are accepted: every entry (``FlatReImCodec``) and the upper
    triangle only (``UpperTriangleReImCodec``), in which case the missing lower
    triangle is filled from its transpose.
    """
    N = _infer_port_count(s_param_dict)

    num_freq = len(frequencies)

    # Check frequency length
    if num_freq < 1:
        raise ValueError("Frequency array must have at least one element.")

    # Initialize S-matrix of shape (nb_f, N, N)
    S = np.zeros((num_freq, N, N), dtype=np.complex64)

    # Fill S-matrix
    for i in range(N):
        for j in range(N):
            name = f"S{i + 1}{j + 1}"
            if f"{name}_real" not in s_param_dict:
                # Upper-triangle-only output: the entry is stored transposed
                name = f"S{j + 1}{i + 1}"

            real = np.array(s_param_dict[f"{name}_real"]).squeeze()
            imag = np.array(s_param_dict[f"{name}_imag"]).squeeze()

            if real.shape[0] != num_freq or imag.shape[0] != num_freq:
                raise ValueError(
                    f"S{i + 1}{j + 1} length mismatch with frequency array."
                )
            S[:, i, j] = real + 1j * imag  # note: frequency as first dimension

    # Create skrf Network object
    ntwk = rf.Network(frequency=frequencies, s=S, f_unit="Hz")

    merged_output = {}
    for i in range(N):
        for j in range(N):
            merged_output[f"S{i + 1}{j + 1}"] = S[:, i, j]

    return N, ntwk, merged_output


def s_param_list_to_network(s_param_list: np.ndarray) -> tuple[int, list[rf.Network]]:
    # Assume s_param_list shape is (batch_size, num_params)
    num_params = s_param_list.shape[1]
    N = int(np.sqrt(num_params // 2))  # number of ports
    logger.debug(f"Number of ports inferred: {N}")
    # Create a network for each sample in the batch
    ntwk_list = []
    for sample in s_param_list:
        S = np.zeros((1, N, N), dtype=np.complex64)  # single frequency point
        for i in range(N):
            for j in range(N):
                real = sample[2 * (i * N + j)]
                imag = sample[2 * (i * N + j) + 1]
                S[0, i, j] = real + 1j * imag
        ntwk = rf.Network(frequency=[1e9], s=S, f_unit="GHz")  # dummy frequency
        ntwk_list.append(ntwk)
    return N, ntwk_list


def single_ended_to_mixed_mode(ntwk: rf.Network) -> rf.Network:
    """
    Converts a 4-port single-ended network to a 2-port mixed-mode network using rf.se2gmm.
    Usually port 1 and 2 are considered differential pair 1, and port 3 and 4 differential pair 2.

    Args:
        ntwk (rf.Network): 4-port single-ended network, converted in place.

    Returns:
        rf.Network: 2-port mixed-mode network.
    """
    ntwk.se2gmm(p=2)
    return ntwk


def plot_diff_s_params_and_k(ntwk: rf.Network):
    """
    Plots the differential S-parameters and coupling factor k for a 4-port single-ended network.

    Args:
        ntwk (rf.Network): 4-port single-ended network.
    """
    # Calculate k
    z = ntwk.z
    ImZ12 = np.imag(z[:, 0, 1])
    ImZ11 = np.imag(z[:, 0, 0])
    ImZ22 = np.imag(z[:, 1, 1])

    with np.errstate(divide="ignore", invalid="ignore"):
        # We use ntwk.f/1e9 to explicitly ensure the X-axis is in GHz
        freq_ghz = ntwk.f
        k_vs_f = np.abs(ImZ12) / np.sqrt(np.abs(ImZ11 * ImZ22))

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Primary Y-Axis (S-parameters)
    ax1.set_xlabel("Frequency")
    ax1.set_ylabel("S-Parameters (dB)")
    # Plotted against the same frequency axis as k below (ntwk.f in Hz), which
    # Network.plot_s_db would not do: it scales the axis to the network's unit.
    ax1.plot(freq_ghz, ntwk.s_db[:, 1, 0], label="Insertion Loss ($S_{d2d1}$)")
    ax1.plot(freq_ghz, ntwk.s_db[:, 0, 0], label="Return Loss ($S_{d1d1}$)")
    ax1.plot(freq_ghz, ntwk.s_db[:, 3, 0], label="Mode Conversion ($S_{c2d1}$)")
    ax1.legend(loc="lower left")

    # Secondary Y-Axis (k)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Coupling Factor ($k$)", color="red")
    # Use freq_ghz here instead of f_scaled
    ax2.plot(freq_ghz, k_vs_f, color="red", linewidth=2, label="Coupling Factor ($k$)")
    ax2.set_ylim(0, 1.1)
    ax2.tick_params(axis="y", labelcolor="red")

    plt.title("Octagon Transformer: Mixed-Mode S-Params & Coupling")
    fig.tight_layout()
    plt.show()


# Chart tokens for saved figures (light surface), shared with the COBRA/ORCA chart palette
_SURFACE = "#fcfcfb"
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_GRIDLINE = "#e1e0d9"
_SERIES = "#2a78d6"


def plot_errors_vs_frequency(profile, path: str, title: str, n_columns: int = 3) -> None:
    """
    Save a small-multiples plot of the test error against frequency, one panel per quantity.

    Each panel shows the median over the test geometries as a line, the middle half of
    them (25th to 75th percentile) as a dark band and the 5th to 95th percentile as a
    light one, on a logarithmic error axis. Written with matplotlib's Agg canvas rather
    than pyplot, so it changes no global backend (the GUI keeps its own) and needs no
    display on a cluster node.

    Args:
        profile (pd.DataFrame): Long table with the columns ``parameter``, ``unit``,
            ``frequency_hz``, ``p5``, ``p25``, ``median``, ``p75`` and ``p95``, one row per
            quantity and frequency point, as written by ``ModelTester``.
        path (str): PNG file to write.
        title (str): Figure title.
        n_columns (int): Panels per row.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    parameters = list(dict.fromkeys(profile["parameter"]))
    n_rows = -(-len(parameters) // n_columns)
    fig = Figure(figsize=(4.2 * n_columns, 3.0 * n_rows + 0.9), dpi=150, facecolor=_SURFACE)
    FigureCanvasAgg(fig)
    axes = fig.subplots(n_rows, n_columns, sharex=True, squeeze=False).ravel()

    for ax, parameter in zip(axes, parameters, strict=False):
        rows = profile[profile["parameter"] == parameter]
        f_ghz = rows["frequency_hz"].to_numpy() / 1e9
        unit = rows["unit"].iloc[0]

        ax.set_facecolor(_SURFACE)
        ax.fill_between(f_ghz, rows["p5"], rows["p95"], color=_SERIES, alpha=0.14, linewidth=0)
        ax.fill_between(f_ghz, rows["p25"], rows["p75"], color=_SERIES, alpha=0.32, linewidth=0)
        ax.plot(f_ghz, rows["median"], color=_SERIES, linewidth=2, solid_capstyle="round")
        if (rows[["p5", "median", "p95"]].to_numpy() > 0).any():
            ax.set_yscale("log")

        kind = "relative error (%)" if unit == "%" else "absolute error"
        ax.set_title(f"{parameter}: {kind}", color=_INK_PRIMARY, fontsize=10, loc="left")
        ax.grid(visible=True, which="major", color=_GRIDLINE, linewidth=0.8, linestyle="-")
        ax.set_axisbelow(True)
        ax.tick_params(colors=_INK_SECONDARY, labelsize=8, which="both")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_GRIDLINE)

    for ax in axes[len(parameters) :]:
        ax.remove()
    # Every panel without one below it carries the frequency axis, also above an empty slot
    for i, ax in enumerate(axes[: len(parameters)]):
        if i + n_columns >= len(parameters):
            ax.xaxis.set_tick_params(labelbottom=True)
            ax.set_xlabel("Frequency (GHz)", color=_INK_SECONDARY, fontsize=9)

    fig.suptitle(title, color=_INK_PRIMARY, fontsize=12, x=0.01, ha="left")
    fig.legend(
        handles=[
            Line2D([], [], color=_SERIES, linewidth=2, label="median"),
            Patch(color=_SERIES, alpha=0.32, label="25th-75th percentile"),
            Patch(color=_SERIES, alpha=0.14, label="5th-95th percentile"),
        ],
        loc="upper right",
        ncols=3,
        frameon=False,
        fontsize=9,
        labelcolor=_INK_PRIMARY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, facecolor=_SURFACE)
