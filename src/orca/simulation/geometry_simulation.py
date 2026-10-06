"""
Simulate one sample of a geometry with Palace, outside the pipeline.

The pipeline simulates many samples through its stages and tables. A consumer such as
COBRA's EM fine-tuning needs the same steps for one parameter set at a time and only
wants the resulting network; :func:`simulate_geometry` is that entry point, so it does
not have to wire the conversion, launch and Touchstone naming together itself.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Any

import skrf as rf

from orca.logger import logger
from orca.simulation.combine_snp_results import touchstone_filename
from orca.simulation.gds_converter import create_palace_model_from_gds
from orca.simulation.launchers import LocalLauncher
from orca.simulation.simulate import patch_palace_config, run_palace

if TYPE_CHECKING:
    from orca.geometry.base_geometry import BaseGeometry

#: The Touchstone variants written after a run, most corrected first. The DC-extrapolated
#: ones exist only when the sweep starts at or below 1 GHz with more than 20 points.
RESULT_PREFERENCE = ("dc_deembedded", "deembedded", "dc", "normal")


class SimulationError(RuntimeError):
    """A geometry sample could not be drawn, meshed or simulated."""


def simulate_geometry(
    geometry: BaseGeometry,
    params: dict[str, Any],
    output_dir: str,
    name: str,
    palace_executable: str = "palace",
    num_processes: int = 1,
    min_element_quality: float = 1e-6,
    save_log: bool = False,
) -> rf.Network:
    """
    Draws, meshes and simulates one sample of a geometry with Palace and returns its network.

    Runs the steps of the pipeline for a single parameter set: :meth:`BaseGeometry.create_gds_file`,
    the gds2palace conversion with the sample's own ports (:meth:`BaseGeometry.ports_for`), the
    mesh quality check of ``GDSConverter``, ORCA's Palace config overrides, and one Palace run on
    this machine. The result is therefore comparable to the training data of the geometry's model.

    Meshing runs in a fresh process, since gmsh must run in a process's main thread (not in a
    GUI worker thread); only the geometry's file paths are sent to it, so a geometry class
    loaded from a file works as well.

    Args:
        geometry (BaseGeometry): The geometry to draw.
        params (dict[str, Any]): The sample's input parameters.
        output_dir (str): Directory for the GDS file, the Palace model and the Touchstone files.
        name (str): Name of the sample, used for the GDS, model and Touchstone file names.
        palace_executable (str): Command that runs Palace (e.g. ``palace`` or an Apptainer call).
        num_processes (int): Number of MPI processes for Palace.
        min_element_quality (float): Meshes whose worst element quality is at or below this are
            not simulated, as in ``GDSConverter``.
        save_log (bool): Write Palace's output to ``palace.log`` in the model directory.

    Returns:
        rf.Network: The most corrected result Palace produced (see :data:`RESULT_PREFERENCE`).

    Raises:
        SimulationError: The parameters are infeasible, the mesh is degenerate, Palace failed,
            or no Touchstone file was written.
    """
    if not geometry.is_feasible(params):
        raise SimulationError(
            f"{type(geometry).__name__} cannot draw {params}: is_feasible rejects them."
        )
    os.makedirs(output_dir, exist_ok=True)
    gds_path = os.path.join(output_dir, f"{name}.gds")
    geometry.create_gds_file(name=name, output_path=gds_path, params=params)

    _, _, config_name, sim_path, data_dir, quality = _create_palace_model(
        geometry_name=name,
        params=params,
        output_dir=output_dir,
        gds_filename=gds_path,
        stackup_xml=geometry.stackup_xml,
        simconfig_filename=geometry.simconfig_filename,
        ports=geometry.ports_for(params),
    )
    # NaN (no volume elements) and -inf (unreadable mesh) fail this check too
    if not quality > min_element_quality:
        raise SimulationError(
            f"The mesh of {name} has a worst element quality of {quality:.3g} (at most "
            f"{min_element_quality:g}); Palace cannot solve it. See {sim_path}."
        )

    patch_palace_config(os.path.join(sim_path, config_name))
    launcher = LocalLauncher()
    succeeded = run_palace(
        sim_path=sim_path,
        data_dir=data_dir,
        result_dir=output_dir,
        cmd=launcher.command(launcher.slots[0], palace_executable, num_processes, config_name),
        touchstone_type="all",
        save_log=save_log,
    )
    if not succeeded:
        raise SimulationError(
            f"The Palace simulation of {name} failed in {sim_path}; the log output above "
            "has the reason."
        )
    return rf.Network(_result_file(output_dir, name, geometry.n_ports))


def _create_palace_model(**kwargs: Any) -> tuple[str, dict[str, Any], str, str, str, float]:
    """:func:`create_palace_model_from_gds` in a fresh process (see :func:`simulate_geometry`)."""
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as executor:
        return executor.submit(create_palace_model_from_gds, **kwargs).result()


def _result_file(output_dir: str, name: str, n_ports: int) -> str:
    """The most corrected Touchstone file written for *name*."""
    candidates = [
        os.path.join(output_dir, touchstone_filename(name, n_ports, variant))
        for variant in RESULT_PREFERENCE
    ]
    for variant, path in zip(RESULT_PREFERENCE, candidates, strict=True):
        if os.path.isfile(path):
            if variant != RESULT_PREFERENCE[0]:
                logger.info(
                    f"Using the {variant} result of {name}: the DC point is only extrapolated "
                    "when the sweep starts at or below 1 GHz with more than 20 points."
                )
            return path
    raise SimulationError(
        f"Palace finished for {name}, but no Touchstone file was written; looked for "
        + ", ".join(os.path.basename(path) for path in candidates)
        + f" in {output_dir}."
    )
