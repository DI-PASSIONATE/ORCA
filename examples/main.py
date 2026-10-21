
import orca
from orca.geometry.presets import TransformerOcta  # or InductorOcta

PLOT = False

# Fixed hyperparameters skip tuning; pass them as ModelTrainer(hyperparameters=...).
hyperparameters = {
    "learning_rate": 0.00037,
    "batch_size": 1024,
    "num_layers": 8,
    "hidden_size": 1805,
    "activation_function": "GELU",
    "basis_degree": 14,
    "epochs": 70,
    "patience": 20,
}


def main():
    # Use predefined geometry from examples
    geometry = TransformerOcta(name="tf_octa_c_ports")
    # geometry = InductorOcta(name="inductor_octa")  # with InductorOcta imported above

    orca_instance = orca.ORCA(
        [
            orca.GDSGenerator(num_samples=4000),
            orca.DRCChecker(),
            orca.GDSConverter(),
            orca.PalaceSimulator(palace_executable="palace"),
            orca.ModelTrainer(
                basis="chebyshev",
                max_epochs=70,
                hyperparameters=hyperparameters,
                allow_tf32=True,
                admittance_weight=1.0,
                above_srf_weight=0.3,
                passivity_weight=1.0,
            ),
            orca.OnnxExporter(),
            orca.ModelTester(plot=PLOT),
        ]
    )

    # seed fixes every random draw of the run (parameter samples, splits, tuning, weights)
    orca_instance.run(geometry=geometry, num_processes=16, seed=40)

if __name__ == "__main__":
    main()
