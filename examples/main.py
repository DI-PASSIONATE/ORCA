import orca

# from orca.geometry.presets.tf_octa_c_ports import TransformerOcta
from orca.geometry.presets.inductor_octa import InductorOcta

PLOT = False

# To run the pipeline on your own geometry, subclass BaseGeometry (name, stackup_xml,
# simconfig_filename, input_parameter_iterator, create_gds_file) and pass an instance
# to orca_instance.run() below; see docs/custom_class.md for a complete example.

hyperparameters = {
    "learning_rate": 0.0005,
    "batch_size": 256,
    "epochs": 15,
    "num_layers": 5,
    "hidden_size": 800,
    "activation_function": "GELU"
}

def main():
    # Use predefined geometry from examples
    try:
        import torch  # optional: only installed with ORCA's "train" extra
        torch.manual_seed(40)
    except ModuleNotFoundError:
        pass
    geometry = InductorOcta()

    orca_instance = orca.ORCA(
        [
            orca.GDSGenerator(num_samples=6000, seed=40),
            orca.DRCChecker(),
            orca.GDSConverter(),
            #orca.PalaceSimulator(palace_executable="apptainer exec ~/Documents/git/palace/palace.sif palace"),
            #orca.ModelTrainer(model=orca.OrcaMLP, hyperparameters=hyperparameters, n_train_samples=1000),
            #orca.OnnxExporter(),
            #orca.ModelTester(),
        ]
    )

    orca_instance.run(geometry=geometry, num_processes=16)

if __name__ == "__main__":
    main()
