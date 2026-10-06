import orca

# from orca.geometry.presets import TransformerOcta
from orca.geometry.presets import InductorOcta


def main():
    # Same geometry (and name) as in simulate.py, which simulates the models this job writes
    # geometry = TransformerOcta(name="tf_octa_c_ports")
    geometry = InductorOcta(name="inductor_octa")

    # The single-node part of the run: layouts, DRC and Palace models. simulate.py picks the models
    # up from output/<geometry name>/palace_sims in a separate multi-node job.
    orca_instance = orca.ORCA(
        [
            orca.GDSGenerator(num_samples=4000),
            orca.DRCChecker(),
            orca.GDSConverter(),
        ]
    )

    # num_processes=None uses all cores of the node. If the job hits its time limit, submit it again
    # (submit.sh): every stage keeps what the previous job finished and only does the rest; the
    # fixed seed gives the missing samples the same parameters as in an uninterrupted run. Pass
    # overwrite=True to a stage to start it from scratch instead.
    orca_instance.run(geometry=geometry, num_processes=None, force_overwrite=True, seed=40)


if __name__ == "__main__":
    main()
