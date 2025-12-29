# Multi-Timescale Model Predictive Control  

This repository contains code and data accompanying the paper **[Multi-Timescale Model Predictive Control for Slow-Fast Systems](https://arxiv.org/abs/2511.14311)**.  

See the **`notebooks/`** folder to reproduce the figures and experiments.
The **`ffmpc_interface/`** folder provides a lightweight wrapper around acados that lets you build your own embedded MTS-MPC solver in just a few lines of code, together with an example showing how to use it.

## Setup 

### 1. Clone the repo

```bash
git clone git@github.com:l-m-schroth/multi-timescale-mpc.git
cd multi-timescale-mpc
```

### 2. (Optional) create a fresh environment

```bash
python -m venv venv          # or: python3 -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the setup script

```bash
pip install -e .
```

### 5. Acados

Our code makes use of **[acados](https://docs.acados.org/)**. If you already have an existing acados installation on your device, try running it. We encountered small issues with different multi-phase solver libraries wrongfully influencing each other. If that happens, install acados from our **[fork](https://github.com/l-m-schroth/acados)**, where these issues were (minimally) addressed. In any case you need to install the python interface in the newly created venv as highlighted in the acados installation guide:

```bash
pip install -e <acados_root>/interfaces/acados_template
```