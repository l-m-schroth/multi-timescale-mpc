"""
Extract masses and inertias of trunk bodies from mujoco.
(The original trunk like system was created in mujoco and then derived as a ODE to ensure reasonable dynamics)
"""
import os
import mujoco
from utils_shared import get_dir

# Function to load a model and extract mass and moments of inertia for a specific body
def extract_mass_and_inertia(model_path, body_name):
    # Load the model
    model = mujoco.MjModel.from_xml_path(model_path)
    
    # Get the body ID
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    
    # Extract mass and moments of inertia
    mass = model.body_mass[body_id]
    inertia = model.body_inertia[body_id]  # Ixx, Iyy, Izz
    return mass, inertia

# List of model file configurations
n_values = [2, 4, 8, 16]
dir = get_dir("src/trunk")
base_path = dir / "archive" / "models_mujoco"
body_name = "actuatedB_1"  # The specific body for which we want data

# Iterate over the models and extract data for the specific body
for n in n_values:
    # Construct the XML file path
    model_path = os.path.join(base_path, f"chain_{n}_links_expanded.xml")
    
    # Extract mass and moments of inertia for the specific body
    mass, inertia = extract_mass_and_inertia(model_path, body_name)
    
    # Print the results
    print(f"Model: chain_{n}_links_expanded.xml")
    print(f"  Body: {body_name}")
    print(f"    Mass: {mass}")
    print(f"    Moments of Inertia: Ixx={inertia[0]}, Iyy={inertia[1]}, Izz={inertia[2]}\n")
