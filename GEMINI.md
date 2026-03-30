# miThermal: LWIR Hyperspectral Image Simulator

miThermal is a high-precision simulator for the acquisition of hyperspectral images in the Long-Wave Infrared (LWIR) range from 3D scenes. It models the propagation of thermal radiation emitted by objects with real physical properties (temperature, spectral emissivity) while considering atmospheric absorption effects.

## 🏗 Project Architecture

The project follows a decoupled architecture with a FastAPI backend and a Gradio frontend.

### 🔙 Backend (`backend/`)
- **Server**: FastAPI application running on `uvicorn`.
- **Core Logic**:
  - `mitsuba_core/`: Integration with **Mitsuba 3** (using `cuda_ad_spectral` variant) for spectral rendering.
  - `atmosphere/`: Modeling of atmospheric absorption and attenuation (CO2, H2O, CH4, O3, etc.).
  - `data_processing/`: Tools for handling hyperspectral data, SPDs, and image generation.
  - `api/`: REST API controllers and services for scene management, rendering, and configuration.
- **Assets**: Contains spectral signatures (`signatures/`), atmospheric reference data (`reference_data/`), and 3D scenes (`mitsuba_scenes/`).

### 🎨 Frontend (`frontend/`)
- **Interface**: Built with **Gradio**, providing an interactive web UI for scene visualization, parameter tuning, and simulation control.
- **Communication**: Interacts with the backend via REST API (configurable via `MITSUBA_API_BASE` environment variable).

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- CUDA-compatible GPU (highly recommended for Mitsuba performance)

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd miThermal

# Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# or
pip install -e .
```

### Running the Project

#### 1. Start the Backend API
```bash
python backend/main.py
```
- API will be available at `http://localhost:8000`
- Interactive documentation at `http://localhost:8000/docs`

#### 2. Start the Frontend UI
```bash
python frontend/main.py
```
- Web interface will be available at `http://localhost:7860`

## 🛠 Development Conventions

- **Code Style**: The project uses `ruff` for linting. Configuration is in `pyproject.toml`.
- **Configuration**: Global settings are managed in `backend/src/config.py`.
- **Output**: All generated data (renders, logs, temp files) is stored in `backend/output/`.
- **Testing**: Unit tests are located in `backend/tests/`. Run them using `pytest`.
- **Notebooks**: Use the `notebooks/` directory for research, prototyping, and validation of physical models.

## 📂 Key Directories

- `backend/src/api/`: API endpoints and business logic.
- `backend/src/mitsuba_core/`: Mitsuba 3 scene generation and rendering.
- `backend/src/atmosphere/`: Atmospheric physics modeling.
- `backend/assets/reference_data/`: Spectral data for gases and materials.
- `backend/assets/signatures/`: ECOSTRESS/JHU spectral library signatures.
- `frontend/src/gradio_interface/`: UI components and event handlers.
- `scripts/`: Utility scripts for batch processing and analysis.

## 🧪 Validation
The simulator's physical accuracy is validated against theoretical models and experimental data. Validation logic and results can be found in `notebooks/validations.ipynb` and related notebooks.
