import orca
from orca.geometry.presets import InductorOcta, TransformerOcta

PLOT = False

# To run the pipeline on your own geometry, subclass BaseGeometry (name, stackup_xml,
# simconfig_filename, input_parameter_iterator, create_gds_file) and pass an instance
# to orca_instance.run() below; see docs/custom_class.md for a complete example.

hyperparameters = {
    "learning_rate": 0.0005,
    "batch_size": 4096,
    "epochs": 60,
    "num_layers": 5,
    "hidden_size": 1024,
    "activation_function": "GELU"
}

def main():
    # Use predefined geometry from examples
    geometry = TransformerOcta(name="transformer_octa")
    geometry = InductorOcta(name="inductor_octa")

    orca_instance = orca.ORCA(
        [
            orca.GDSGenerator(num_samples=6000),
            orca.DRCChecker(),
            orca.GDSConverter(),
            #orca.PalaceSimulator(palace_executable="apptainer exec ~/Documents/git/palace/palace.sif palace"),
            #orca.ModelTrainer(model=orca.OrcaMLP, hyperparameters=hyperparameters, n_train_samples=1000),
            #orca.OnnxExporter(),
            #orca.ModelTester(),
        ]
    )

    # seed fixes every random draw of the run (parameter samples, splits, tuning, weights)
    orca_instance.run(geometry=geometry, num_processes=16, seed=40)

if __name__ == "__main__":
    main()
