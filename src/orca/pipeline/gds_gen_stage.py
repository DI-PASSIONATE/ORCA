import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from typing import TYPE_CHECKING, Any

import pandas as pd
import tqdm

from orca.logger import logger
from orca.pipeline.pipeline_stage import PipelineStage
from orca.pipeline.resume import append_row, resume_table

if TYPE_CHECKING:
    from orca.geometry.base_geometry import BaseGeometry
    from orca.geometry.input_parameters import InputParameterIterator
    from orca.pipeline.context import PipelineContext


class GDSGenerator(PipelineStage):
    """
    Pipeline stage for generating GDS files from trained models.

    Existing layouts are kept: a run finds the layouts of an earlier, possibly
    aborted run in the GDS table and only draws and lays out the missing sample
    indices. With the same run seed (``ORCA.run(seed=...)``) the draws are the same as
    in a fresh run.

    Args:
        num_samples (int): Number of parameter combinations to draw and lay out.
        overwrite (bool): Delete the layouts of earlier runs and draw all samples anew.
    """

    def __init__(
        self, num_samples: int = 1000, overwrite: bool = False, plot_coverage: bool = True
    ):
        """
        Args:
            num_samples (int): Number of parameter samples, and thus GDS layouts, to generate.
            plot_coverage (bool): Save a pair plot of the laid-out parameter combinations,
                over the draws rejected as infeasible, to geometries/<name>_coverage.png.
            overwrite (bool): Delete the geometry folder first instead of keeping the layouts
                an earlier run left there.
        """
        super().__init__(name="GDS Generator", index=0)
        self.num_samples = num_samples
        self.plot_coverage = plot_coverage
        self.overwrite = overwrite

    def run(
        self,
        context: "PipelineContext",
        progress_callback: Callable[[str, int, int, str], None] | None = None,
    ) -> "PipelineContext":
        geometry: BaseGeometry = context.geometry
        cpu_cores: int = context.num_processes
        output_dir = context.geometry_dir
        gds_csv = context.gds_csv_path
        logger.info(
            f"Starting GDS generation for {self.num_samples} samples using {cpu_cores} CPU cores."
        )

        if self.overwrite and os.path.exists(output_dir):
            import shutil

            shutil.rmtree(output_dir)

        os.makedirs(output_dir, exist_ok=True)

        # Tell the input parameter iterator the number of samples to generate; draws
        # the geometry cannot build are rejected there and, for the random
        # strategy, redrawn, so the requested number of layouts is met.
        iterator = geometry.input_parameter_iterator
        iterator.set_sample_count(
            self.num_samples, seed=context.seed, feasible=geometry.is_feasible
        )

        # Layouts an earlier run finished are kept. Their parameters are not compared with
        # this run's draws (an unseeded run would then redraw everything); the later stages
        # compare parameters and redo a sample whose layout changed.
        names = [f"{geometry.name}_{i}.gds" for i in range(self.num_samples)]
        existing = resume_table(
            gds_csv,
            pd.DataFrame({"name": names}),
            ["name", *iterator.input_names],
            lambda row: os.path.exists(os.path.join(output_dir, row["name"])),
        )
        if existing:
            logger.info(
                f"Reusing {len(existing)} of {self.num_samples} layouts from an earlier run in "
                f"{output_dir}."
            )

        futures = []
        failed = 0
        with ProcessPoolExecutor(max_workers=cpu_cores) as executor:
            # Create cpu_cores processes to generate GDS files in parallel. The iterator is
            # walked through reused indices too, so each index gets the same draw as in a
            # fresh run with the same seed.
            for i, input_params in enumerate(geometry.input_iterator):
                if i >= self.num_samples:
                    break
                if names[i] in existing:
                    continue
                if not futures:
                    # The DRC tables describe the previous set of layouts; drop them so the
                    # converter does not mistake them for a check of the new ones.
                    for path in (context.drc_csv_path, context.drc_report_path):
                        if os.path.exists(path):
                            os.remove(path)

                # Pass geometry class and name separately to avoid pickling the whole instance (with locks)
                future = executor.submit(
                    GDSGenerator._generate_gds_file,
                    geometry.create_gds_file,
                    names[i],
                    output_dir,
                    input_params,
                )
                futures.append(future)

            # Print progress bar using tqdm and call progress_callback
            for i, future in enumerate(
                tqdm.tqdm(
                    as_completed(futures), total=len(futures), desc="GDS Generation"
                )
            ):
                try:
                    _gds_path, name, params = future.result()

                    # Save instance name + input parameters to CSV
                    append_row(gds_csv, {"name": name} | params)
                except BrokenProcessPool:
                    # A worker died (e.g. the calling script re-ran the pipeline on
                    # import); nothing else will finish, so abort instead of skipping
                    raise
                except Exception as e:  # noqa: BLE001 - one bad sample must not abort the batch
                    failed += 1
                    logger.debug(f"Worker task failed: {e}")
                finally:
                    if progress_callback:
                        progress_callback(
                            self.name,
                            i + 1,
                            len(futures),
                            f"GDS Generation Progress: {i + 1}/{len(futures)}",
                        )

        drawn = len(existing) + len(futures) - failed
        if iterator.n_rejected:
            logger.info(
                f"{iterator.n_rejected} infeasible parameter combinations were rejected by "
                f"{type(geometry).__name__}.is_feasible and redrawn."
            )
        if failed or drawn < self.num_samples:
            # A geometry that raises for a draw its is_feasible accepted, or a grid
            # strategy with infeasible points, yields fewer layouts than requested;
            # that should not go unnoticed.
            logger.warning(
                f"GDS generation completed: {drawn} of {self.num_samples} requested layouts "
                f"drawn; {failed} draws failed in create_gds_file (see the debug log)."
            )
        else:
            logger.info(f"GDS generation completed: {drawn} layouts drawn.")
        if self.plot_coverage:
            self._plot_coverage(context, iterator)
        context.gds_csv = gds_csv
        return context

    @staticmethod
    def _plot_coverage(context: "PipelineContext", iterator: "InputParameterIterator") -> None:
        """Plot which parts of the parameter box the layouts in the GDS table cover."""
        if not os.path.exists(context.gds_csv_path) or not iterator.input_names:
            return
        layouts = pd.read_csv(context.gds_csv_path)
        rejected = pd.DataFrame(iterator.rejected, columns=iterator.input_names)
        # Imported here: matplotlib is only needed once the layouts are done
        from orca.utils.postprocessing import plot_parameter_coverage

        path = context.gds_coverage_plot_path
        title = f"{context.geometry.name}: parameter coverage ({len(layouts)} layouts"
        if iterator.n_rejected:
            title += f", {iterator.n_rejected} draws rejected as infeasible"
        try:
            plot_parameter_coverage(
                layouts, rejected, iterator.parameters, path, title=title + ")"
            )
        except Exception as e:  # noqa: BLE001 - a failed plot must not fail the layouts
            logger.warning(f"Could not plot the parameter coverage: {e}")
            return
        logger.info(f"Parameter coverage plotted to {path}.")

    @staticmethod
    def _generate_gds_file(
        gds_method: Callable, name: str, output_dir: str, params: dict[str, Any]
    ) -> tuple[str, str, dict[str, Any]]:
        """
        Static method to generate a GDS file given a geometry class, name, and parameters.
        This is used for multiprocessing to avoid pickling issues with instance methods.

        Args:
            gds_method (Callable): The geometry's create_gds_file(name, output_path, params).
            name (str): The name of the GDS file to create.
            output_dir (str): The directory to save the GDS file.
            params (dict[str, Any]): The input parameters for the geometry.

        Returns:
            str: Path to the created GDS file.
        """
        output_path = os.path.join(output_dir, name)
        path = gds_method(name, output_path, params)
        return path, name, params
