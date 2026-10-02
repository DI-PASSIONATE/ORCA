import os

import orca
from orca.geometry.presets import InductorOcta

# from orca.geometry.presets import TransformerOcta


def main():
    # Same geometry (and name) as in simulate.py, so the results are found
    # geometry = TransformerOcta(name="tf_octa_c_ports")
    geometry = InductorOcta(name="inductor_octa")

    run_dir = os.path.join(os.getcwd(), "output", geometry.name)
    result_dir = os.path.join(run_dir, "results")
    # One folder per job: the simulation's run record and earlier models are never overwritten
    base_dir = os.path.join(run_dir, f"training_{os.environ.get('SLURM_JOB_ID', 'local')}")

    orca_instance = orca.ORCA(
        [
            orca.ModelTrainer(
                n_fold_cv=3,
                n_trials=100,
                batch_sizes=[1024, 2048, 4096, 8192, 16384],
                tuning_max_epochs=30,
                basis="chebyshev",
                # Stop tuning after 16 h, leaving the rest of the 24 h limit for the final
                # training, the export and the test. A trial still running at that point is
                # pruned after its current epoch.
                tuning_timeout=16 * 3600,
                max_epochs=200,
            ),
            orca.OnnxExporter(),
            orca.ModelTester(),
        ]
    )

    orca_instance.run(geometry=geometry, base_dir=base_dir, result_dir=result_dir, seed=40)


if __name__ == "__main__":
    main()
