import orca
from orca.geometry.presets import TransformerOcta  # or InductorOcta

PLOT = False

# To run the pipeline on your own geometry, subclass BaseGeometry (name, stackup_xml,
# simconfig_filename, input_parameter_iterator, create_gds_file) and pass an instance
# to orca_instance.run() below; see docs/custom_class.md for a complete example.

# Fixed hyperparameters skip tuning; pass them as ModelTrainer(hyperparameters=...).
# hyperparameters = {
#     "learning_rate": 0.0004,
#     "batch_size": 2048,
#     "num_layers": 8,
#     "hidden_size": 1775,
#     "activation_function": "GELU",
#     "basis_degree": 14,
# }

def main():
    # Use predefined geometry from examples
    geometry = TransformerOcta(name="transformer_octa")
    # geometry = InductorOcta(name="inductor_octa")  # with InductorOcta imported above

    orca_instance = orca.ORCA(
        [
            orca.GDSGenerator(num_samples=4000),
            orca.DRCChecker(),
            orca.GDSConverter(),
            orca.PalaceSimulator(palace_executable="palace"),
            orca.ModelTrainer(
                basis="chebyshev",
                max_epochs=300,
                allow_tf32=True,
                admittance_weight=0.5,
                above_srf_weight=0.6,
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
