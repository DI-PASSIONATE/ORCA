import json
import os
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING

import pandas as pd
import tqdm

from orca.geometry.drc import GRID_NM, DRCResult, PortContact, check_gds_file, port_contacts
from orca.logger import logger
from orca.pipeline.pipeline_stage import PipelineStage
from orca.simulation.simulate import read_simconfig

if TYPE_CHECKING:
    from orca.pipeline.context import PipelineContext


class DRCChecker(PipelineStage):
    """
    Pipeline stage that snaps the generated GDS files to the manufacturing grid
    and checks them against the IHP SG13G2 design rules.

    Layout code produces off-grid vertices (path extrusion on angled segments)
    and, for some parameter combinations, geometry that no fab would accept:
    traces narrower than the rules allow, vias clipped by a miter, folded
    polygons. Simulating those wastes compute and teaches the model shapes that
    cannot be built. This stage repairs what can be repaired in place (the grid)
    and records the rest, so the conversion stage only picks up clean layouts.

    It also checks that every port marker touches the metals its port connects, with
    each layout's own ports (``BaseGeometry.ports_for``). A marker that misses its metal by a few nanometres
    still simulates, as an open circuit, so the result describes another circuit
    than the parameters say. Such layouts are always left out.

    Outputs, in the geometry folder: ``<name>_drc_report.csv`` with the
    violation counts of every layout, and ``<name>_drc.csv``, the parameter
    table of the layouts that passed, in the same layout as the GDS table.
    The rules are those of :mod:`orca.geometry.drc`.
    """

    def __init__(
        self,
        grid_nm: int = GRID_NM,
        snap_to_grid: bool = True,
        drop_violations: bool = True,
        check_ports: bool = True,
    ):
        """
        Args:
            grid_nm (int): Manufacturing grid in nanometres. SG13G2 uses 5 nm.
            snap_to_grid (bool): Move every vertex onto the grid and write the GDS file
                back before checking, so off-grid vertices are repaired instead of reported.
            drop_violations (bool): Leave layouts with violations out of the parameter
                table the later stages use. Off, every layout goes on and the violations
                are only reported.
            check_ports (bool): Check that every port marker touches the metals its port
                connects, with the ports of each sample. Layouts with an open port are left out
                whatever ``drop_violations`` says: their simulation would describe a
                circuit with that port disconnected.
        """
        super().__init__(name="DRC Checker", index=1)
        if grid_nm <= 0:
            raise ValueError(f"grid_nm must be positive, got {grid_nm}.")
        self.grid_nm = grid_nm
        self.snap_to_grid = snap_to_grid
        self.drop_violations = drop_violations
        self.check_ports = check_ports

    def run(
        self,
        context: "PipelineContext",
        progress_callback: Callable[[str, int, int, str], None] | None = None,
    ) -> "PipelineContext":
        gds_csv = context.gds_csv_path
        if not os.path.exists(gds_csv):
            logger.error(
                f"No GDS parameter table found at {gds_csv}. The GDS Generator stage was "
                "skipped or produced no layouts; nothing to check."
            )
            return context

        gds_data = pd.read_csv(gds_csv)
        gds_dir = os.path.dirname(gds_csv)
        ports = self._port_contacts(context, gds_data) if self.check_ports else {}
        logger.info(
            f"Starting DRC of {len(gds_data)} GDS files on a {self.grid_nm} nm grid "
            f"using {context.num_processes} CPU cores."
        )

        results: dict[str, DRCResult] = {}
        with ProcessPoolExecutor(max_workers=context.num_processes) as executor:
            futures = {
                executor.submit(
                    check_gds_file,
                    os.path.join(gds_dir, name),
                    self.grid_nm,
                    self.snap_to_grid,
                    ports.get(name, ()),
                ): name
                for name in gds_data["name"]
            }
            for i, future in enumerate(
                tqdm.tqdm(as_completed(futures), total=len(futures), desc="DRC Check")
            ):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as e:  # noqa: BLE001 - one unreadable file must not abort the batch
                    logger.error(f"DRC of {name} failed with error: {e}")
                    results[name] = DRCResult(violations={"error": 1})
                finally:
                    if progress_callback:
                        progress_callback(
                            self.name,
                            i + 1,
                            len(futures),
                            f"Checked {i + 1} of {len(futures)} GDS files.",
                        )

        report = pd.DataFrame(
            [
                {
                    "name": name,
                    "snapped_vertices": results[name].snapped_vertices,
                    "violations": results[name].total,
                    "rules": ";".join(
                        f"{rule}:{count}"
                        for rule, count in sorted(
                            (results[name].violations | results[name].port_findings).items()
                        )
                    ),
                }
                for name in gds_data["name"]
            ]
        )
        report.to_csv(context.drc_report_path, index=False)

        clean = gds_data["name"].map(lambda name: results[name].clean)
        connected = gds_data["name"].map(lambda name: results[name].ports_connected)
        passed = gds_data[clean] if self.drop_violations else gds_data[connected]
        passed.to_csv(context.drc_csv_path, index=False)

        summary: Counter[str] = Counter()
        for result in results.values():
            summary.update(result.violations)
            summary.update(result.port_findings)
        self._log_open_ports(results, context.drc_report_path)
        self._log_summary(
            len(gds_data),
            int(clean.sum()),
            summary,
            int(report["snapped_vertices"].sum()),
            context.drc_report_path,
        )

        context.drc_csv = context.drc_csv_path
        context.drc_summary = dict(sorted(summary.items()))
        return context

    @staticmethod
    def _port_contacts(
        context: "PipelineContext", gds_data: pd.DataFrame
    ) -> dict[str, tuple[PortContact, ...]]:
        """
        Per layout, the metals each of its ports has to touch: the sample's ports
        (``BaseGeometry.ports_for``), with layer numbers from the stackup.
        """
        geometry = context.geometry
        datatypes = tuple(
            read_simconfig(geometry.simconfig_filename)["saved_values"].get("purpose", [0])
        )
        # Most samples share one port set; look each distinct set up once
        by_ports: dict[str, tuple[PortContact, ...]] = {}
        contacts = {}
        for row in gds_data.to_dict("records"):
            name = row.pop("name")
            ports = geometry.ports_for(row)
            key = json.dumps(ports, sort_keys=True)
            if key not in by_ports:
                by_ports[key] = port_contacts(ports, geometry.stackup_xml, datatypes)
            contacts[name] = by_ports[key]
        return contacts

    @staticmethod
    def _log_open_ports(results: dict[str, DRCResult], report_path: str) -> None:
        open_ports = [name for name, result in results.items() if not result.ports_connected]
        if open_ports:
            logger.error(
                f"{len(open_ports)} of {len(results)} layouts have port markers that do not "
                "touch their metal; they would simulate as open circuits and were dropped. "
                f"Examples: {', '.join(open_ports[:5])}. Details are in {report_path}."
            )

    def _log_summary(
        self, total: int, n_clean: int, summary: Counter[str], snapped: int, report_path: str
    ) -> None:
        if self.snap_to_grid:
            logger.info(f"Snapped {snapped} off-grid vertices to the {self.grid_nm} nm grid.")
        if n_clean == total:
            logger.info(f"DRC completed: all {total} layouts are clean.")
            return
        rules = ", ".join(f"{rule} x{count}" for rule, count in summary.most_common())
        verb = "dropped" if self.drop_violations else "kept, drop_violations is off"
        logger.warning(
            f"DRC completed: {n_clean} of {total} layouts are clean; "
            f"{total - n_clean} with violations {verb}. Findings: {rules}. "
            f"Per-layout counts are in {report_path}."
        )
