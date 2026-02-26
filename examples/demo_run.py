import sys
import os
import numpy as np

# Add root to path so we can import vol2pc without installation
# Since we are in examples/, we need to go up one level
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vol2pc import registry
from vol2pc.core.pipeline import Reader, run_pipeline
from vol2pc.core.types import Volume
from vol2pc.config import Config
from vol2pc import register_builtin_plugins

# Define a Dummy Reader for testing
class DummyReader(Reader):
    def read(self, path: str, **kwargs) -> Volume:
        print(f"Generating dummy volume for {path}...")
        # Create a simple 32x32x32 volume
        # Gradient along X
        dims = (32, 32, 32)
        x = np.linspace(0, 1, dims[2])
        y = np.linspace(0, 1, dims[1])
        z = np.linspace(0, 1, dims[0])
        
        # Create meshgrid
        # Note: meshgrid order='ij' -> (dims[0], dims[1], dims[2]) if indexed by z, y, x
        Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
        
        # Function: sphere at center
        # Center is 0.5, 0.5, 0.5
        dist = np.sqrt((X-0.5)**2 + (Y-0.5)**2 + (Z-0.5)**2)
        data = np.exp(-10 * dist**2).astype(np.float32)
        
        return Volume(data=data, spacing=(0.1, 0.1, 0.1))

# Register it
# Ensure builtins are registered too if we rely on them (e.g. normalize)
register_builtin_plugins()
registry.register_reader("dummy", DummyReader)

def main():
    # Construct config in memory
    config_dict = {
        "io": {
            "reader": "dummy",
            "path": "synthetic_data" # Name doesn't matter for dummy reader
        },
        "preprocess": [
            {
                "name": "normalize",
                "method": "minmax"
            }
        ],
        "sampling": {
            "name": "gradient",
            "n_points": 5000
        },
        "export": {
            "writer": "ply",
            "path": "output_demo.ply"
        }
    }
    
    cfg = Config.from_dict(config_dict)
    
    print("Starting Demo Pipeline...")
    pc = run_pipeline(None, None, cfg)
    
    print(f"Pipeline Finished.")
    print(f"Generated Point Cloud with {pc.xyz.shape[0]} points.")
    print(f"Output saved to {config_dict['export']['path']}")
    
    # Verify file exists
    if os.path.exists("output_demo.ply"):
        print("Verification: Output file exists.")
    else:
        print("Verification: Output file missing!")

if __name__ == "__main__":
    main()
