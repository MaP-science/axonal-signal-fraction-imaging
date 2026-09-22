import sys
from asfimaging.pipeline.ASFpipeline import ASFpipeline



def train_RF_regressor(config_name):
    # Run the ASF pipeline for training and saving a regressor based on synthetic data 
    ASFImaging = ASFpipeline(config_name)
    trained_regressor = ASFImaging.train_synthetic()
    return trained_regressor

if __name__ == "__main__":
    # Generate the scheme for the HCP dataset
    config_name = sys.argv[1]

    RF_regressor = train_RF_regressor(config_name)

    print(f"Trained a RF regressor and saved it at the model path")

# python -m scripts.train "configs/hcp_noise_free.json"