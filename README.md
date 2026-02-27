# VolSampler: Volume to Point Cloud Sampler

A flexible, plugin-based framework for converting volumetric data (e.g., VTI, Medical Imaging) into Point Clouds using various sampling strategies.

## Features

*   **Modular Pipeline**: Read -> Preprocess -> Sample -> Export.
*   **Plugin System**: Easily extendable with custom Readers, Stages, Samplers, and Writers.
*   **Advanced Sampling**: Includes Uniform, Gradient-based, and **Wavelet-based** sampling.
*   **CLI Support**: Full command-line interface for batch processing.

## Installation

```bash
# Clone the repository
git clone https://github.com/Loveof1ife7/VolSampler-volume-to-point-cloud-sampler.git
cd VolSampler

# Install dependencies
pip install -r requirements.txt
# OR
pip install .
```

## Quick Start

### 1. Run a Demo

```bash
python examples/demo_run.py
```

### 2. Using CLI

Process a volume using a configuration file:

```bash
python -m vol2pc.cli run -i datasets/data.vti -c config/config.yaml -o output.ply
```

### 3. Configuration Example

Create a `config.yaml` file:

```yaml
io:
  reader: vti  # Built-in VTI reader

preprocess:
  - name: normalize
    method: minmax

sampling:
  name: wavelet      # Use Wavelet Sampler
  n_points: 50000    # Number of points
  level: 3           # Decomposition level
  wavelet: haar      # Wavelet type
  device: cuda       # processing device

export:
  writer: ply        # Export to PLY
```

## Supported Components

*   **Readers**: `vti`
*   **Preprocess**: `normalize` (minmax, percentile)
*   **Samplers**: 
    *   `uniform`: Random uniform sampling.
    *   `gradient`: Magnitude-weighted gradient sampling.
    *   `wavelet`: Multi-level wavelet coefficient sampling (supports large volumes via chunking).
*   **Writers**: `ply`

## Project Structure

```text
vol2pc/
├── core/           # Core types and pipeline logic
├── io/             # Data readers
├── preprocess/     # Preprocessing stages
├── sampling/       # Sampling algorithms (Wavelet, Gradient, etc.)
├── export/         # Data writers
└── cli.py          # Command Line Interface
```

## Developing Plugins

To add a new sampler, create a class inheriting from `Sampler` and register it:

```python
from vol2pc.core.pipeline import Sampler
from vol2pc.registry import register_sampler

class MySampler(Sampler):
    def sample(self, vol, cfg):
        # ... implementation ...
        return point_cloud

register_sampler("my_sampler", MySampler)
```
