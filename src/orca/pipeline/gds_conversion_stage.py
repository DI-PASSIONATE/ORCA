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
    """

    def __init__(self, timeout: float = 60.0, overwrite: bool = False):
        """
        Args:
            timeout (float): Maximum seconds a single GDS conversion may take
                before its worker is killed and the sample is skipped.
            overwrite (bool): Delete the Palace model folder first instead of keeping
                the models an earlier run left there.
        """
        super().__init__(name="GDS Converter", index=2)
        self.timeout = timeout
        self.overwrite = overwrite

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
            # Print progress bar using tqdm
            for i, future in enumerate(
                tqdm.tqdm(
                    as_completed(futures), total=len(futures), desc="GDS Conversion"
                )
            ):
                name = futures[future]
                try:
                    geo_name, params, config_name, sim_path, data_dir = future.result()
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

        context.palace_csv = palace_csv
        return context
