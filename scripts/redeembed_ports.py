# ruff: noqa: INP001, T201 - a standalone maintenance script that reports on stdout
"""
Recompute the port de-embedding of existing ORCA results with the fixed algorithm.

Before 2026-10-12, ``port_deembedding`` cascaded every port's negative series inductance at
port 1 and none at the other ports. This script rewrites every ``*_deembedded.sNp`` file it
finds below a directory laid out like ORCA's ``output/``::

    <root>/.../inductor_octa/results/inductor_octa_<i>_dc.s3p
    <root>/.../inductor_octa/results/inductor_octa_<i>_dc_deembedded.s3p   <- rewritten
    <root>/.../inductor_octa/geometries/inductor_octa_<i>.gds
    <root>/.../inductor_octa/geometries/inductor_octa.csv
    <root>/.../transformer_octa/...                                         (same layout)

The Palace folders and their ``port_information.json`` files are usually gone by then, so the
port sizes are rebuilt the way gds2palace's ``add_ports`` measures them: the bounding box of the
port polygon in the stored GDS and the z gap between the port's two metals in the stackup, for
the sample's own ports (``BaseGeometry.ports_for``). The de-embedding itself is ORCA's
``port_deembedding``, applied to the file the de-embedded one was made from (``_dc_deembedded``
from ``_dc``, ``_deembedded`` from the plain file). Only de-embedded files that already exist are
rewritten; nothing new is created.

Usage:
    uv run python scripts/redeembed_ports.py output            # rewrite
    uv run python scripts/redeembed_ports.py output --dry-run  # only report the changes
"""

import argparse
import csv
import os
import re
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Any

import numpy as np
import skrf as rf

from orca import BaseGeometry
from orca.geometry.presets import InductorOcta, TransformerOcta
from orca.simulation.combine_snp_results import port_deembedding

PresetClass = type[InductorOcta] | type[TransformerOcta]

# Results folder name prefix -> geometry class. The geometry name is set to the folder name,
# so a TransformerOcta run saved as "transformer_octa" is found as well as "tf_octa_c_ports".
GEOMETRY_CLASSES: dict[str, PresetClass] = {
    "inductor": InductorOcta,
    "transformer": TransformerOcta,
    "tf_octa": TransformerOcta,
}

DEEMBEDDED_RE = re.compile(r"^(?P<sample>.+?_\d+)(?P<dc>_dc)?_deembedded\.s(?P<n>\d+)p$")

# A rewritten file that differs from the existing one by less than this is left untouched
UNCHANGED_TOL = 1e-12


@dataclass
class Job:
    geometry_name: str
    geometry_class: PresetClass
    params: dict[str, Any]
    gds_filename: str
    source_filename: str
    target_filename: str


def geometry_class_for(folder_name: str) -> PresetClass | None:
    for prefix, cls in GEOMETRY_CLASSES.items():
        if folder_name.startswith(prefix):
            return cls
    return None


def read_parameter_table(run_dir: str, geometry_name: str) -> dict[str, dict[str, Any]]:
    """Input parameters per sample ("inductor_octa_12"), from the geometries or results CSV."""
    table: dict[str, dict[str, Any]] = {}
    for sub in ("results", "geometries"):  # geometries last: it lists every drawn sample
        csv_filename = os.path.join(run_dir, sub, f"{geometry_name}.csv")
        if not os.path.isfile(csv_filename):
            continue
        with open(csv_filename, newline="") as f:
            for row in csv.DictReader(f):
                name = row.pop("name")
                sample = re.sub(r"(_dc)?(_deembedded)?\.(gds|s\d+p)$", "", name)
                table[sample] = {key: float(value) for key, value in row.items()}
    return table


def find_jobs(root: str) -> list[Job]:
    jobs: list[Job] = []
    for dirpath, dirnames, _ in os.walk(root):
        dirnames.sort()
        if os.path.basename(dirpath) != "results":
            continue
        run_dir = os.path.dirname(dirpath)
        geometry_name = os.path.basename(run_dir)
        cls = geometry_class_for(geometry_name)
        if cls is None:
            continue
        params_table = read_parameter_table(run_dir, geometry_name)
        for filename in sorted(os.listdir(dirpath)):
            match = DEEMBEDDED_RE.match(filename)
            if match is None:
                continue
            sample = match["sample"]
            source = f"{sample}{match['dc'] or ''}.s{match['n']}p"
            gds_filename = os.path.join(run_dir, "geometries", f"{sample}.gds")
            missing = [
                what
                for what, ok in (
                    (source, os.path.isfile(os.path.join(dirpath, source))),
                    (gds_filename, os.path.isfile(gds_filename)),
                    (f"parameters of {sample}", sample in params_table),
                )
                if not ok
            ]
            if missing:
                print(f"SKIP {filename}: missing {', '.join(missing)}", file=sys.stderr)
                continue
            jobs.append(
                Job(
                    geometry_name=geometry_name,
                    geometry_class=cls,
                    params=params_table[sample],
                    gds_filename=gds_filename,
                    source_filename=os.path.join(dirpath, source),
                    target_filename=os.path.join(dirpath, filename),
                )
            )
    return jobs


_geometries: dict[tuple[PresetClass, str], BaseGeometry] = {}
_stackups: dict[str, Any] = {}


def port_information(job: Job) -> dict[str, Any]:
    """The port_information.json content gds2palace would have written for this sample."""
    from gds2palace import gds_reader, stackup_reader

    key = (job.geometry_class, job.geometry_name)
    if key not in _geometries:
        _geometries[key] = job.geometry_class(name=job.geometry_name)
    geometry = _geometries[key]
    if geometry.stackup_xml not in _stackups:
        _, _, metals = stackup_reader.read_substrate(geometry.stackup_xml)
        _stackups[geometry.stackup_xml] = metals
    metals = _stackups[geometry.stackup_xml]

    ports = geometry.ports_for(job.params)
    by_layer = {port["source_layernum"]: port for port in ports}
    # gds2palace prints on every read
    with open(os.devnull, "w") as null, redirect_stdout(null):
        polygons = gds_reader.read_gds(
            job.gds_filename,
            list(by_layer),
            purposelist=[0],
            metals_list=metals,
        ).polygons

    # Like add_ports: one entry per port polygon; a later polygon of the same port wins
    sizes: dict[int, dict[str, Any]] = {}
    for poly in polygons:
        port = by_layer.get(poly.layernum)
        if port is None:
            continue
        if port.get("target_layername") is not None:
            raise NotImplementedError(f"In-plane port {port['portnumber']} of {job.gds_filename}")
        from_metal = metals.getbylayername(port["from_layername"])
        to_metal = metals.getbylayername(port["to_layername"])
        lower, upper = sorted((from_metal, to_metal), key=lambda m: m.zmin)
        sizes[port["portnumber"]] = {
            "portnumber": port["portnumber"],
            "length": upper.zmin - lower.zmax,
            "width": max(poly.xmax - poly.xmin, poly.ymax - poly.ymin),
        }
    missing = sorted({port["portnumber"] for port in ports} - set(sizes))
    if missing:
        raise ValueError(f"No polygon for port(s) {missing} in {job.gds_filename}")
    return {"unit": 1e-6, "ports": list(sizes.values())}


def redeembed(job: Job, dry_run: bool) -> tuple[str, str, float]:
    """Rewrite one de-embedded file. Returns (target, status, max |ΔS| to the old file)."""
    try:
        info = port_information(job)
        results_dir = os.path.dirname(job.target_filename)
        with tempfile.TemporaryDirectory(dir=results_dir, prefix=".redeembed_") as tmp:
            source = os.path.join(tmp, os.path.basename(job.source_filename))
            shutil.copyfile(job.source_filename, source)
            port_deembedding(source, True, info)
            stem, ext = os.path.splitext(source)
            written = f"{stem}_deembedded{ext}"

            new = rf.Network(written)
            old = rf.Network(job.target_filename)
            if new.s.shape != old.s.shape or not np.allclose(new.f, old.f):
                delta = float("inf")
            else:
                delta = float(np.max(np.abs(new.s - old.s)))
            if delta < UNCHANGED_TOL:
                return job.target_filename, "unchanged", delta
            if not dry_run:
                os.replace(written, job.target_filename)
            return job.target_filename, "would rewrite" if dry_run else "rewritten", delta
    except Exception as exc:  # noqa: BLE001 - report and continue with the other samples
        return job.target_filename, f"FAILED: {exc!r}", float("nan")


def _run(args: tuple[Job, bool]) -> tuple[str, str, float]:
    return redeembed(*args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("root", help="Directory to search, e.g. output/")
    parser.add_argument("--dry-run", action="store_true", help="Report, do not write")
    parser.add_argument("-j", "--jobs", type=int, default=None, help="Worker processes")
    args = parser.parse_args()

    jobs = find_jobs(args.root)
    print(f"Found {len(jobs)} de-embedded files below {args.root}")
    counts: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for target, status, delta in pool.map(
            _run, [(job, args.dry_run) for job in jobs], chunksize=16
        ):
            kind = status.split(":")[0]
            counts[kind] = counts.get(kind, 0) + 1
            if kind != "unchanged":
                print(f"{status:>13}  max|dS|={delta:.3e}  {os.path.relpath(target, args.root)}")
    print(", ".join(f"{n} {kind}" for kind, n in sorted(counts.items())))
    if counts.get("FAILED"):
        sys.exit(1)


if __name__ == "__main__":
    main()
