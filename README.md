# PointCloud2Texture

## Overview
PointCloud2Texture is a Python-based tool for applying textures to 3D meshes using point clouds. It utilizes six projection planes to generate high-quality textures for 3D reconstructions.

## Features
- Processes 3D meshes and point clouds (.ply format)
- Generates textured meshes using six projection planes
- Supports debugging options to save intermediate results
- Simple command-line interface for execution

## Installation
### **1. Clone the repository**
```sh
git clone https://github.com/**********/pointcloud2texture.git
cd pointcloud2texture
```

### **2. Create a virtual environment (optional but recommended)**
```sh
python3 -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

### **3. Install dependencies**
```sh
pip install -r requirements.txt
```

### **4. Install the package in editable mode**
```sh
pip install -e .
```

## Usage
### **Basic Execution**
To apply textures to a mesh, use the following command:
```sh
run_texturing --mesh data/maoka/mesh_level1.ply --pcd data/maoka/point_cloud.ply
```

### **Enable Debug Mode (Optional)**
```sh
run_texturing --mesh data/maoka/mesh_level1.ply --pcd data/maoka/point_cloud.ply --debug
```

### **Output Files**
The processed textures and masks will be saved in the `outputs/` directory.
```
├── outputs
│   └── 20250312
│       └── maoka
│           ├── textured_mesh.mtl
│           ├── textured_mesh.obj
│           ├── textured_mesh_0.png
│           ├── textured_mesh_1.png
│           ├── textured_mesh_2.png
│           ├── textured_mesh_3.png
│           ├── textured_mesh_4.png
│           └── textured_mesh_5.png
```
## Project Structure
```
pointcloud2texture/
├── configs/            # Configuration files
├── data/               # Input point clouds and meshes
├── models/             # Pre-trained models (if applicable)(For future version)
├── outputs/            # Output directory for textures and masks
├── scripts/            # Execution main script
├── src/                # Source code
│   ├── texturing/      # Main processing module
│   ├── __init__.py
│   ├── six_projection_planes.py
│   ├── utils.py
├── requirements.txt    # Dependencies
├── setup.py            # Installation script
└── README.md           # Documentation
```

## Troubleshooting
If you encounter module import errors, ensure you run the script from the project root and use the correct Python environment:
```sh
PYTHONPATH=$(pwd) python3 scripts/run_texturing.py --mesh data/maoka/mesh_level1.ply --pcd data/maoka/point_cloud.ply
```

