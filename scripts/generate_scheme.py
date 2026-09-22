# This script generates the scheme for the HCP dataset

import os
import numpy as np
import json
import pandas as pd
import sys
import pickle
from asfimaging.preprocessing import generate_scheme as gs

  


if __name__ == "__main__":
    # Load config
    config_name = sys.argv[1]

    with open(f"{config_name}", "r") as f:
        config = json.load(f)

    pickle_path = config['data_paths']['scheme']
    if os.path.exists(pickle_path):
        print(f"Scheme already exists at {pickle_path}. Loading scheme.")
        with open(pickle_path, "rb") as f:
            scheme = pickle.load(f)
        print(f"Scheme loaded from {pickle_path}")
        print(scheme.head())
        sys.exit(0)
    
    

    # Generate scheme
    scheme_generator = gs.SchemeGenerator(config)
    scheme = scheme_generator.generate_scheme()

    # Save scheme
    path = scheme_generator.save(scheme)

    print(f"Scheme generated and saved to {path}")


