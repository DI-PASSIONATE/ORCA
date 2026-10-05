import os
import shutil
from collections.abc import Callable
from concurrent.futures import as_completed
from typing import TYPE_CHECKING

import pandas as pd
import tqdm
from pebble import ProcessPool

from orca.logger import logger
from orca.pipeline.pipeline_stage import PipelineStage
from orca.pipeline.resume import append_row, resume_table
from orca.simulation.gds_converter import create_palace_model_from_gds
from orca.simulation.simulate import read_simconfig

if TYPE_CHECKING:
    from orca.geometry.base_geometry import BaseGeometry
    from orca.pipeline.context import PipelineContext

#: Columns of the Palace table between the sample name and its parameters.
PALACE_PATH_COLUMNS = ["data_dir", "sim_path", "config_name"]


class GDSConverter(PipelineStage):
    """
    Pipeline stage for converting GDS files to Palace-compatible format.

    Each conversion runs in a worker process with a time limit. gmsh can loop
    forever on degenerate geometry (self-intersecting boundary curves that its
    2D mesher never recovers from), and a plain process pool would then wait
    for that worker indefinitely. A worker that exceeds ``timeout`` is killed
    and replaced, and the sample is logged and skipped like a failed mesh.

    Palace models of an earlier, possibly aborted run are kept: a layout is only
    converted if the Palace table has no row for it with the same parameters, or
    its Palace config is gone.

    Every new mesh is checked for flat, inverted or corrupt elements, which Palace cannot
    solve (``HasPositiveFiniteDiagonal(...) is false``): such a sample is left out of
    the Palace table instead of failing on the cluster. The worst element quality of
    each mesh is written to ``<name>_mesh_report.csv`` next to the table.
    """

    def __init__(
        self,
        timeout: float = 180.0,
        overwrite: bool = False,
        min_element_quality: float = 1e-6,
        warn_element_quality: float = 1e-4,
    ):
        """
        Args:
            timeout (float): Maximum seconds a single GDS conversion may take
                before its worker is killed and the sample is skipped. It only guards
                against gmsh hanging: a large inductor at refined_cellsize 2 already
                takes up to ~50 s, and a sample cut off here is lost silently.
            overwrite (bool): Delete the Palace model folder first instead of keeping
                the models an earlier run left there.
            min_element_quality (float): Meshes whose worst element quality (gmsh's
                minSICN: 1 regular, 0 flat) is at or below this are left out. Flat
                elements come out at 0 up to rounding, so the default only catches those.
            warn_element_quality (float): Meshes with a worst element quality below this
                are kept but reported. Meshes that Palace has solved had 6e-4 and more.
        """
        super().__init__(name="GDS Converter", index=2)
        self.timeout = timeout
        self.overwrite = overwrite
        self.min_element_quality = min_element_quality
        self.warn_element_quality = warn_element_quality

    def run(
        self,
        context: "PipelineContext",
        progress_callback: Callable[[str, int, int, str], None] | None = None,
    ) -> "PipelineContext":
        geometry: BaseGeometry = context.geometry
        cpu_cores: int = context.num_processes
        base_dir: str = context.base_dir
        # The DRC stage leaves the table of the layouts that passed next to the GDS
        # table; the GDS Generator deletes it whenever it adds layouts, so a present
        # table belongs to the current layouts even when the DRC stage ran in an
        # earlier session.
        gds_csv = context.gds_csv_path
        if os.path.exists(context.drc_csv_path):
            gds_csv = context.drc_csv_path
            logger.info(f"Converting the layouts that passed DRC, listed in {gds_csv}.")
        elif not os.path.exists(gds_csv):
            logger.error(
                f"No GDS parameter table found at {gds_csv}. The GDS Generator stage was "
                "skipped or produced no layouts; nothing to convert."
            )
            return context
        output_dir = context.palace_sim_dir
        palace_csv = context.palace_csv_path

        if self.overwrite and os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        os.makedirs(output_dir, exist_ok=True)

        gds_data = pd.read_csv(gds_csv)
        total = len(gds_data)

        # Keep the models an earlier run finished for the same parameters
        param_columns = [column for column in gds_data.columns if column != "name"]
        converted = resume_table(
            palace_csv,
            gds_data,
            ["name", *PALACE_PATH_COLUMNS, *param_columns],
            lambda row: os.path.exists(os.path.join(row["sim_path"], row["config_name"])),
        )
        gds_data = gds_data[~gds_data["name"].isin(converted)]
        if converted:
            logger.info(
                f"Reusing {len(converted)} of {total} Palace models from an earlier run in "
                f"{output_dir}."
            )
        # A layout converted before with other parameters leaves a stale mesh and results
        # in its folder; start it clean.
        for name in gds_data["name"]:
            sample_dir = os.path.join(output_dir, f"{os.path.splitext(name)[0]}_data")
            if os.path.isdir(sample_dir):
                shutil.rmtree(sample_dir)

        logger.info(
            f"Starting GDS conversion for {len(gds_data)} files using {cpu_cores} CPU cores."
        )

        with ProcessPool(
            max_workers=cpu_cores
        ) as pool:
            futures = {}
            gds_dir = os.path.dirname(gds_csv)
            for _, row in gds_data.iterrows():
                # CSV layout:
                # name,input_winding_diameter,output_winding_diameter,center_displacement,bottom_linewidth,upper_linewidth
                # everything after name is input parameters
                name = row["name"]  # GDS path
                gds_path = os.path.join(gds_dir, name)
                params = row.to_dict()
                del params["name"]

                # Submit GDS conversion tasks
                future = pool.schedule(
                    create_palace_model_from_gds,
                    kwargs={
                        "geometry_name": name,
                        "params": params,
                        "output_dir": base_dir,
                        "gds_filename": gds_path,
                        "stackup_xml": geometry.stackup_xml,
                        "simconfig_filename": geometry.simconfig_filename,
                        "show_mesh_results": False,
                        "ports": geometry.ports_for(params),
                    },
                    timeout=self.timeout,
                )
                futures[future] = name

            # Collect finished results
            qualities: dict[str, float] = {}
            # Print progress bar using tqdm
            for i, future in enumerate(
                tqdm.tqdm(
                    as_completed(futures), total=len(futures), desc="GDS Conversion"
                )
            ):
                name = futures[future]
                try:
                    geo_name, params, config_name, sim_path, data_dir, quality = future.result()
                    qualities[name] = quality
                    if quality <= self.min_element_quality:
                        # Palace would stop on this mesh; keep it out of the table
                        continue
                    # Save input parameters to CSV
                    append_row(
                        palace_csv,
                        {
                            "name": geo_name,
                            "data_dir": data_dir,
                            "sim_path": sim_path,
                            "config_name": config_name,
                        }
                        | params,
                    )
                except TimeoutError:
                    logger.error(
                        f"GDS conversion of {name} exceeded {self.timeout:g} s "
                        "(gmsh did not finish meshing); skipping this sample."
                    )
                except Exception as e:  # noqa: BLE001 - one bad sample must not abort the batch
                    logger.error(f"GDS conversion failed for {name} with error: {e}")
                finally:  # and call progress_callback even on failure
                    if progress_callback:
                        progress_callback(
                            self.name,
                            i + 1,
                            len(futures),
                            f"Converted {i + 1} of {len(futures)} GDS files.",
                        )

            logger.info("GDS conversion completed.")

        self._report_mesh_quality(qualities, context)
        context.palace_csv = palace_csv
        return context

    def _report_mesh_quality(self, qualities: dict[str, float], context: "PipelineContext") -> None:
        """Write the worst element quality of each new mesh and warn about poor ones."""
        if not qualities:
            return
        report_path = context.mesh_report_path
        report = pd.DataFrame(
            {"name": list(qualities), "worst_element_quality": list(qualities.values())}
        )
        if os.path.exists(report_path):
            # Meshes of earlier runs stay listed; a sample meshed again replaces its row
            earlier = pd.read_csv(report_path)
            report = pd.concat([earlier[~earlier["name"].isin(qualities)], report])
        report.to_csv(report_path, index=False)

        flat = sorted(n for n, q in qualities.items() if q <= self.min_element_quality)
        poor = sorted(
            n for n, q in qualities.items()
            if self.min_element_quality < q < self.warn_element_quality
        )
        cell_size = read_simconfig(context.geometry.simconfig_filename)["saved_values"].get(
            "refined_cellsize"
        )
        hint = (
            f"If this happens often, refined_cellsize = {cell_size:g} µm in the simcfg may be too "
            "coarse for these layouts."
            if cell_size is not None
            else "If this happens often, the mesh may be too coarse for these layouts."
        )
        if flat:
            logger.warning(
                f"{len(flat)} of {len(qualities)} meshes contain flat, inverted or corrupt "
                f"elements (worst quality at or below {self.min_element_quality:g}; -inf: gmsh "
                f"cannot read the mesh back) and were left out, as Palace cannot solve them. "
                f"{hint} Samples: {', '.join(flat[:10])}"
                f"{' ...' if len(flat) > 10 else ''}. Qualities are in {report_path}."
            )
        if poor:
            logger.warning(
                f"{len(poor)} of {len(qualities)} meshes have poorly shaped elements (worst "
                f"quality below {self.warn_element_quality:g}); they were kept. {hint} "
                f"Samples: {', '.join(poor[:10])}{' ...' if len(poor) > 10 else ''}."
            )
