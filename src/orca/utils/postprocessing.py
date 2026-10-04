from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import skrf as rf

from orca.logger import logger


def differential_impedance(
    ntwk: rf.Network, pairs: Sequence[tuple[int, int]], shorted: Sequence[int] = ()
) -> np.ndarray:
    """
    The Z-matrix of port pairs driven differentially, each by a floating source.

    The ports in ``shorted`` are AC-grounded (a center tap in differential operation);
    every other port that is in no pair is left open. Entry ``(a, b)`` is the voltage
    across pair ``a`` per current driven through pair ``b`` (into its first port, out of
    its second): ``Z[pa, pb] - Z[pa, nb] - Z[na, pb] + Z[na, nb]`` of the network with
    the shorted ports removed.

    Args:
        ntwk (rf.Network): The single-ended network.
        pairs: ``(positive, negative)`` port indices, zero-based, per differential port.
        shorted: Zero-based indices of the ports to short.

    Returns:
        np.ndarray: Complex, shape ``(n_freq, len(pairs), len(pairs))``.
    """
    kept = [port for port in range(ntwk.nports) if port not in shorted]
    position = {port: i for i, port in enumerate(kept)}
    # Shorting a port is dropping its row and column of Y; the inverse is then the
    # Z-matrix with that port grounded and the rest open. A pseudo-inverse, since a
    # winding without a path to ground has a singular Y (its common mode is undefined),
    # while its differential impedance, orthogonal to that mode, is not.
    z = np.linalg.pinv(ntwk.y[:, kept][:, :, kept])
    z_diff = np.empty((len(ntwk.f), len(pairs), len(pairs)), dtype=complex)
    for a, (pa, na) in enumerate(pairs):
        for b, (pb, nb) in enumerate(pairs):
            i, j, k, m = position[pa], position[na], position[pb], position[nb]
            z_diff[:, a, b] = z[:, i, k] - z[:, i, m] - z[:, j, k] + z[:, j, m]
    return z_diff


def inductor_parameters(
    ntwk: rf.Network, ends: tuple[int, int] = (0, 1), shorted: Sequence[int] = ()
) -> dict[str, np.ndarray]:
    """
    Differential inductance, resistance, quality factor and self-resonance of an inductor.

    Args:
        ntwk (rf.Network): The single-ended network.
        ends: Zero-based ports at the two ends of the winding.
        shorted: Ports to AC-ground, such as a center tap.

    Returns:
        dict: ``L`` [nH], ``R`` [Ohm] and ``Q`` over frequency, and ``srf_f`` [GHz],
        where the reactance first turns capacitive (NaN if it stays inductive).
    """
    z = differential_impedance(ntwk, [ends], shorted)[:, 0, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.imag(z) / (2 * np.pi * ntwk.f) * 1e9
        Q = np.imag(z) / np.real(z)
    return {
        "L": L,
        "R": np.real(z),
        "Q": Q,
        "srf_f": np.array(_first_inductive_to_capacitive(ntwk.f / 1e9, np.imag(z))),
    }


def transformer_parameters(
    ntwk: rf.Network,
    primary: tuple[int, int],
    secondary: tuple[int, int],
    shorted: Sequence[int] = (),
) -> dict[str, np.ndarray]:
    """
    Inductance, resistance and quality factor of both windings of a transformer, their
    coupling factor, and the primary's self-resonance, all differential.

    Args:
        ntwk (rf.Network): The single-ended network.
        primary: Zero-based ports at the two ends of the primary winding.
        secondary: The same for the secondary winding.
        shorted: Ports to AC-ground, such as the center taps.

    Returns:
        dict: ``Lp``, ``Ls`` [nH], ``Rp``, ``Rs`` [Ohm], ``Qp``, ``Qs`` and ``k`` over
        frequency, and ``srf_f`` [GHz] of the primary (NaN if it stays inductive).
    """
    z = differential_impedance(ntwk, [primary, secondary], shorted)
    omega = 2 * np.pi * ntwk.f
    zp, zs, zm = z[:, 0, 0], z[:, 1, 1], z[:, 0, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "Lp": np.imag(zp) / omega * 1e9,
            "Ls": np.imag(zs) / omega * 1e9,
            "Rp": np.real(zp),
            "Rs": np.real(zs),
            "Qp": np.imag(zp) / np.real(zp),
            "Qs": np.imag(zs) / np.real(zs),
            "k": np.abs(np.imag(zm)) / np.sqrt(np.abs(np.imag(zp) * np.imag(zs))),
            "srf_f": np.array(_first_inductive_to_capacitive(ntwk.f / 1e9, np.imag(zp))),
        }


def _first_inductive_to_capacitive(freq_ghz: np.ndarray, reactance: np.ndarray) -> float:
    """Where a reactance first falls from positive to negative, interpolated linearly; NaN if never."""
    cross = np.flatnonzero((reactance[:-1] > 0) & (reactance[1:] <= 0))
    if cross.size == 0:
        return float("nan")
    i = cross[0]
    f0, f1 = freq_ghz[i], freq_ghz[i + 1]
    x0, x1 = reactance[i], reactance[i + 1]
    return float(f0 - x0 * (f1 - f0) / (x1 - x0))


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


def plot_electrical_parameters(
    frequencies: np.ndarray,
    reference: dict[str, np.ndarray],
    predicted: dict[str, np.ndarray] | None = None,
    title: str = "",
) -> None:
    """
    Show each electrical parameter over frequency, the reference against a prediction.

    One panel per curve; scalar parameters (such as ``srf_f``) are listed in the title
    and marked as a vertical line. Shown with pyplot, for interactive use.

    Args:
        frequencies (np.ndarray): Frequencies in Hz.
        reference (dict): Parameters of the reference, as a geometry's
            ``electrical_parameters`` returns them.
        predicted (dict | None): The same for the prediction.
        title (str): Figure title.
    """
    curves = [name for name, value in reference.items() if np.ndim(value) == 1]
    scalars = [name for name, value in reference.items() if np.ndim(value) == 0]
    if not curves:
        return
    freq = frequencies / 1e9
    n_columns = min(3, len(curves))
    n_rows = -(-len(curves) // n_columns)
    fig, axes = plt.subplots(
        n_rows, n_columns, figsize=(4.8 * n_columns, 3.4 * n_rows), squeeze=False
    )
    labels = [
        f"{name}: {float(reference[name]):.3g}"
        + (f" (predicted {float(predicted[name]):.3g})" if predicted and name in predicted else "")
        for name in scalars
    ]
    fig.suptitle("  |  ".join([title, *labels]) if labels else title, fontsize=13)

    for ax, name in zip(axes.flat, curves, strict=False):
        ax.plot(freq, reference[name], color="black", lw=2, label="Reference")
        if predicted is not None and name in predicted:
            ax.plot(freq, predicted[name], color="tab:blue", ls="--", label="Predicted")
        for scalar in scalars:
            if np.isfinite(reference[scalar]):
                ax.axvline(float(reference[scalar]), color="red", ls=":", lw=1)
        ax.set_title(name)
        ax.set_xlabel("Frequency [GHz]")
        ax.grid(True, alpha=0.3)
    axes.flat[0].legend()
    for ax in list(axes.flat)[len(curves) :]:
        ax.set_visible(False)

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


#: Draws a geometry rejected: context for the layouts, so a recessive neutral
_REJECTED = "#bdbbb3"


def _histogram_edges(grid, low: float, high: float, log: bool, n_bins: int = 30) -> np.ndarray:
    """
    Bin edges for a histogram of one parameter, equally wide in the axis' own scale.

    For a parameter with a grid of values, every edge lies halfway between two values,
    so each bin holds whole values and the bars do not alternate with how many grid
    values happen to fall into them.
    """
    forward = np.log10 if log else (lambda x: np.asarray(x, dtype=float))
    if grid is None or len(grid) < 2:
        if high <= low:
            return np.array([low - 0.5, high + 0.5]) if not log else np.array([low / 1.1, high * 1.1])
        return np.geomspace(low, high, n_bins + 1) if log else np.linspace(low, high, n_bins + 1)
    t = forward(np.asarray(grid, dtype=float))
    midpoints = (t[1:] + t[:-1]) / 2
    outer = np.concatenate(([t[0] - (t[1] - t[0]) / 2], midpoints, [t[-1] + (t[-1] - t[-2]) / 2]))
    if len(grid) > n_bins:
        targets = np.linspace(outer[0], outer[-1], n_bins + 1)
        outer = np.unique(outer[np.abs(outer[:, None] - targets[None, :]).argmin(axis=0)])
    return 10**outer if log else outer


def plot_parameter_coverage(
    layouts, rejected, parameters: dict, path: str, title: str
) -> None:
    """
    Save a pair plot of the parameter combinations that were laid out.

    One panel per pair of parameters shows every layout as a dot, over the draws the
    geometry's feasibility check rejected in grey, so gaps in the coverage and the
    regions that cannot be built are told apart. The diagonal holds a histogram of
    each parameter. Axes span each parameter's declared range, outlined in every panel
    so samples moved onto the boundary of the box sit on the outline; log-sampled
    parameters get a log axis.

    Args:
        layouts (pd.DataFrame): One row per layout, one column per parameter.
        rejected (pd.DataFrame): One row per rejected draw; may be empty.
        parameters (dict[str, GeometryParameter]): The declared parameters, in order.
        path (str): PNG file to write.
        title (str): Figure title.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    names = list(parameters)
    n = len(names)
    log = {name: getattr(parameters[name], "sampling", "uniform") == "log" for name in names}
    bounds = {name: parameters[name].bounds for name in names}

    def limits(name: str) -> tuple[float, float]:
        low, high = bounds[name]
        if log[name]:
            pad = (high / low) ** 0.04
            return low / pad, high * pad
        pad = 0.04 * (high - low) or 0.5
        return low - pad, high + pad

    def log_axis(axis) -> None:
        # Plain numbers (30, 50, 100) instead of matplotlib's 3x10^1 mathtext
        axis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.set_minor_formatter(NullFormatter())

    def style(ax) -> None:
        ax.set_facecolor(_SURFACE)
        ax.tick_params(colors=_INK_SECONDARY, labelsize=7, which="both")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_GRIDLINE)

    size = 2.2 if n > 1 else 4.0
    fig = Figure(figsize=(size * n + 0.4, size * n + 0.9), dpi=150, facecolor=_SURFACE)
    FigureCanvasAgg(fig)
    axes = fig.subplots(n, n, squeeze=False)

    for row, y_name in enumerate(names):
        for col, x_name in enumerate(names):
            ax = axes[row][col]
            if col > row:
                ax.remove()
                continue
            style(ax)
            if log[x_name]:
                ax.set_xscale("log")
                log_axis(ax.xaxis)
            ax.set_xlim(*limits(x_name))

            if row == col:  # histogram of the layouts along this parameter
                values = layouts[x_name].to_numpy(dtype=float)
                edges = _histogram_edges(
                    getattr(parameters[x_name], "grid_values", None), *bounds[x_name],
                    log=log[x_name],
                )
                counts, _ = np.histogram(values, bins=edges)
                # Density in the axis' own scale, so bins of unequal width compare fairly
                scale = np.log10(edges) if log[x_name] else edges
                ax.bar(
                    edges[:-1], counts / np.diff(scale), width=np.diff(edges), align="edge",
                    color=_SERIES, edgecolor=_SURFACE, linewidth=1,
                )
                ax.set_yticks([])
                ax.spines["left"].set_visible(False)
                ax.set_title(x_name, color=_INK_PRIMARY, fontsize=9, loc="left")
            else:  # layouts over rejected draws, for one pair of parameters
                if log[y_name]:
                    ax.set_yscale("log")
                    log_axis(ax.yaxis)
                ax.set_ylim(*limits(y_name))
                if len(rejected):
                    ax.scatter(
                        rejected[x_name], rejected[y_name], s=2, color=_REJECTED,
                        linewidths=0, rasterized=True,
                    )
                ax.scatter(
                    layouts[x_name], layouts[y_name], s=3, color=_SERIES, alpha=0.7,
                    linewidths=0, rasterized=True,
                )
                (x_low, x_high), (y_low, y_high) = bounds[x_name], bounds[y_name]
                ax.add_patch(
                    Rectangle(
                        (x_low, y_low), x_high - x_low, y_high - y_low, fill=False,
                        edgecolor=_INK_SECONDARY, linewidth=0.6, alpha=0.5,
                    )
                )
            # Tick labels only on the outer panels
            if row != n - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(x_name, color=_INK_SECONDARY, fontsize=8)
            if col == 0 and row > 0:
                ax.set_ylabel(y_name, color=_INK_SECONDARY, fontsize=8)
            elif row != col:
                ax.tick_params(labelleft=False)

    fig.suptitle(title, color=_INK_PRIMARY, fontsize=12, x=0.01, ha="left")
    handles = [Line2D([], [], marker="o", linestyle="", color=_SERIES, label="layouts")]
    if len(rejected):
        handles.append(
            Line2D([], [], marker="o", linestyle="", color=_REJECTED, label="rejected as infeasible")
        )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    # In the empty upper triangle, clear of the title
    fig.legend(
        handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.93), frameon=False,
        fontsize=9, labelcolor=_INK_PRIMARY,
    )
    fig.savefig(path, facecolor=_SURFACE)
