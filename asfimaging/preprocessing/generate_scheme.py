import numpy as np
from pathlib import Path
import pandas as pd
from asfimaging.shtools import sphericalharmonics as sh



class SchemeGenerator:
    """
    Generates acquisition schemes from bvals and bvecs.
    """

    def __init__(self, config: dict):

        self.config = config

    
    def load_bvals_bvecs(self, bvals_path, bvecs_path):
        bvals = np.loadtxt(bvals_path)
        bvecs = np.loadtxt(bvecs_path)

        print(f"Loaded bvals from {bvals_path} with shape {bvals.shape}")
        print(f"Loaded bvecs from {bvecs_path} with shape {bvecs.shape}")

            
        return bvals, bvecs


    def check_data_requirements(self, bvals, bvecs):
        if bvals.shape[0] != bvecs.shape[1]:
            raise ValueError("Number of bvals does not match number of bvecs")

        if np.any(bvals < 0):
            raise ValueError("bvals cannot be negative")

        
        if np.unique(bvals >= 4000).shape[0] < 2:
            raise ValueError("At least two unique high b-values are required for a valid scheme")
                
        vec_needed_for_sh_order = len(sh.compute_lm_ordervector(self.config["real_data_acquisition"]['sh_order']))
        unique_bvals = np.unique(bvals[bvals >= 4000])

        if len(np.where(bvals == unique_bvals[0])[0]) < vec_needed_for_sh_order and len(np.where(bvals == unique_bvals[1])[0]) < vec_needed_for_sh_order:
            raise ValueError(f"Not enough diffusion-weighted directions for the chosen SH order. At least {vec_needed_for_sh_order} are required in each shell.")
        
        return True


    def check_scheme_validity(self, scheme):
        # Check if the scheme is valid for the given sh_order
        sh_order = self.config["real_data_acquisition"]['sh_order']
        LM = sh.compute_lm_ordervector(sh_order) 
        bvalues = scheme.b.values
        delta = scheme.d
        Delta = scheme.D
                
        print(f"Unique b-values in the scheme: {np.unique(bvalues)}",
              f"\nNumber of directions for each unique b-value: {[np.sum(bvalues == bval) for bval in np.unique(bvalues)]}")
        # Directions in each shell should be greater than the number of SH coefficients for the given sh_order
        for bval in np.unique(bvalues)>=4000e6:
            num_directions = scheme[scheme.b == bval].dir.values
            if num_directions < len(LM):
                raise ValueError(f"Not enough directions for b-value {bval}. At least {len(LM)} are required for SH order {sh_order}.")


        return True
    
    def parameters_for_scheme_generation(self, delta, Delta):
        if delta is None or Delta is None:
            print("Warning: delta and Delta are not defined. Scheme will be generated without considering diffusion time.")
            return 0, 0
        else:
            return delta, Delta

        
    def tolarance(self, bvals):
        # Define a tolerance for matching bvals and bvecs
        tol_bval = self.config['real_data_acquisition']['tolerance_bval']
        tol_b0 = self.config['real_data_acquisition']['tolerance_b0']

        bvals[(bvals >= self.config['real_data_acquisition']['high_bmin'] - tol_bval) & (bvals <= self.config['real_data_acquisition']['high_bmin'] + tol_bval)] = self.config['real_data_acquisition']['high_bmin']
        bvals[(bvals >= self.config['real_data_acquisition']['high_bmax'] - tol_bval) & (bvals <= self.config['real_data_acquisition']['high_bmax'] + tol_bval)] = self.config['real_data_acquisition']['high_bmax']

        bvals[(bvals >= -tol_b0) & (bvals <= tol_b0)] = 0 * 1e6

        return bvals
    

    def create_scheme(self, bvecs, bvals):

        if np.all(bvals < 1e6):
            print("Warning: bvals are in units of s/mm^2. Converting to s/m^2.")
            b_vals = bvals * 1e6
    
        # Shell index is determined by the unique b-values, direction index is determined by the unique bvecs for each shell
        shell_index = np.zeros_like(bvals, dtype=int)
        direction_index = np.zeros_like(bvals, dtype=int)
        unique_bvals = np.unique(bvals)
        for i, bval in enumerate(unique_bvals):
            idxs = np.where(bvals == bval)[0]
            shell_index[idxs] = i
            unique_bvecs = np.unique(bvecs[:, idxs], axis=1)
            for j, bvec in enumerate(unique_bvecs.T):
                idxs_dir = np.where((bvecs[:, idxs] == bvec[:, np.newaxis]).all(axis=0))[0]
                direction_index[idxs[idxs_dir]] = j

        
        # Generate the scheme using dmipy
        scheme = pd.DataFrame({
        "x": bvecs[:, 0],
        "y": bvecs[:, 1],
        "z": bvecs[:, 2],
        "te": np.ones_like(bvals) * self.config['real_data_acquisition']['te'],
        "d": np.ones_like(bvals) * self.config['real_data_acquisition']['delta'],
        "D": np.ones_like(bvals) * self.config['real_data_acquisition']['Delta'],
        "b": bvals,
        "dshell": shell_index,
        "dir": direction_index,
        "type": ["pgse"] * len(bvals)
        })

        return scheme


    def generate_scheme(self):

        # Read the bvaecs and bvalues files from config 
        bvals_path = self.config['data_paths']['bvals']
        bvecs_path = self.config['data_paths']['bvecs']
        b_values, b_vectors = self.load_bvals_bvecs(bvals_path, bvecs_path)

        
        # Check if the data meets the requirements for scheme generation
        if not self.check_data_requirements(b_values, b_vectors):
            raise ValueError("Data does not meet the requirements for scheme generation")
        
        # Apply tolarance to bvals
        b_values = self.tolarance(b_values)
        
        # Generate the scheme from the bvals and bvecs
        scheme = self.create_scheme(b_values, b_vectors)

        # Check if the generated scheme is valid for the given sh_order
        if not self.check_scheme_validity(scheme):
            raise ValueError("Generated scheme is not valid for the given SH order, Try lowering the SH order or adjusting the b-values in the acquisition scheme.")
        
        return scheme


    def save(self, scheme):

        output_dir = Path(self.config['data_paths']['scheme'])
        output_dir.mkdir(parents=True, exist_ok=True)

        path = output_dir 

        scheme.to_pickle(path)
        print(f"Scheme saved to {path}")
        return path

