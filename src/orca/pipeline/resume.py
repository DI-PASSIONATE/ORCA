"""Parameter tables that survive an aborted run, and resuming from them.

The sample-producing stages (GDS generation, GDS conversion, Palace simulation)
record every finished sample as one row in a CSV table, appended right after the
sample's output file is written. A row is therefore the commit marker of its
sample: a run killed half-way (a Slurm walltime, a crash) leaves a table that
lists exactly the samples that are complete, and the next run only has to do
the rest.

On resume a stage keeps a row only if its sample is still wanted, its output
file still exists, and its parameter values equal those of the stage's input.
The last check catches layouts that were drawn anew with different parameters
under the same name, whose old meshes and results would otherwise be reused.
"""

import os
import shutil
from collections.abc import Callable

import numpy as np
import pandas as pd

from orca.logger import logger


def read_table(path: str, key: str = "name") -> pd.DataFrame | None:
    """
    Reads a stage's parameter table.

    Lines a killed run left half-written are skipped, and when a sample appears
    twice the later row wins.

    Args:
        path (str): Path to the CSV table.
        key (str): Column holding the sample name.

    Returns:
        pd.DataFrame | None: The table, or None if it does not exist or has no rows.
    """
    if not os.path.exists(path):
        return None
    try:
        table = pd.read_csv(path, on_bad_lines="skip", float_precision="round_trip")
    except pd.errors.EmptyDataError:
        return None
    if table.empty or key not in table.columns:
        return None
    return table.drop_duplicates(subset=key, keep="last")


def write_table(table: pd.DataFrame, path: str) -> None:
    """
    Writes a table so that an abort leaves either the old or the new file, never half of one.

    Args:
        table (pd.DataFrame): The rows to write.
        path (str): Path to the CSV table.
    """
    tmp_path = f"{path}.tmp"
    table.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def append_row(path: str, row: dict) -> None:
    """
    Appends one finished sample to a table, writing the header if the table is new.

    Args:
        path (str): Path to the CSV table.
        row (dict): Column values of the sample, in table column order.
    """
    exists = os.path.exists(path)
    pd.DataFrame([row]).to_csv(path, mode="a" if exists else "w", header=not exists, index=False)


def matching_names(existing: pd.DataFrame, expected: pd.DataFrame, key: str = "name") -> set[str]:
    """
    Names whose row in ``existing`` holds the same values as in ``expected``.

    Every column of ``expected`` other than ``key`` is compared; numbers within a
    relative tolerance of 1e-9, anything else as text. A missing value never matches.

    Args:
        existing (pd.DataFrame): Rows written by an earlier run.
        expected (pd.DataFrame): Rows the current run would write.
        key (str): Column holding the sample name.

    Returns:
        set[str]: Names of the samples that can be reused.
    """
    columns = [column for column in expected.columns if column != key]
    merged = expected[[key, *columns]].merge(
        existing[[key, *columns]], on=key, how="inner", suffixes=("_new", "_old")
    )
    match = np.ones(len(merged), dtype=bool)
    for column in columns:
        new, old = merged[f"{column}_new"], merged[f"{column}_old"]
        new_num = pd.to_numeric(new, errors="coerce")
        old_num = pd.to_numeric(old, errors="coerce")
        numeric = (new_num.notna() & old_num.notna()).to_numpy()
        close = np.isclose(
            new_num.fillna(0).to_numpy(float), old_num.fillna(0).to_numpy(float), rtol=1e-9, atol=0.0
        )
        same_text = ((new.astype(str) == old.astype(str)) & new.notna() & old.notna()).to_numpy()
        match &= np.where(numeric, close, same_text)
    return set(merged.loc[match, key])


def resume_table(
    path: str,
    expected: pd.DataFrame,
    columns: list[str],
    output_exists: Callable[[pd.Series], bool],
    key: str = "name",
) -> set[str]:
    """
    Prepares a stage's table for resuming and returns the samples that are already done.

    A row of the existing table is kept if its sample is in ``expected`` with the same
    values and ``output_exists`` confirms its output file. The table is rewritten with
    only those rows, which also drops a line cut off by a killed run; if that leaves
    rows out, the previous table is first copied to ``<path>.bak``. The caller then
    appends a row for each sample it finishes.

    Args:
        path (str): Path to the stage's CSV table.
        expected (pd.DataFrame): Rows this run would write, reduced to the key and the
            columns to compare (parameter values; not paths, which may differ).
        columns (list[str]): All columns this run writes, in order. An existing table with
            other columns belongs to a different setup and is not resumed.
        output_exists (Callable[[pd.Series], bool]): Whether the output file of an existing
            row is present.
        key (str): Column holding the sample name.

    Returns:
        set[str]: Names of the samples that do not have to be produced again.

    Raises:
        ValueError: If the existing table has different columns than this run writes.
    """
    existing = read_table(path, key)
    if existing is None:
        if os.path.exists(path):
            os.remove(path)
        return set()

    if set(existing.columns) != set(columns):
        raise ValueError(
            f"Cannot resume from {path}: it has the columns {list(existing.columns)}, but this "
            f"run writes {columns}. Enable overwrite on the stage or use another output folder."
        )

    done = matching_names(existing, expected, key)
    by_name = existing.set_index(key, drop=False)
    done = {name for name in done if output_exists(by_name.loc[name])}
    kept = existing.loc[existing[key].isin(done), columns]
    if len(kept) < len(existing):
        shutil.copy2(path, f"{path}.bak")
        logger.warning(
            f"{len(existing) - len(kept)} of {len(existing)} rows in {path} are left out: their "
            "sample is no longer part of this run, was built with other parameters, or its output "
            "file is missing. Samples of this run among them are produced again; the old table is "
            f"kept as {path}.bak."
        )
    write_table(kept, path)
    return done
