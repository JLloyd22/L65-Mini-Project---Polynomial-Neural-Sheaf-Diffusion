# Heterogeneous Sheaf (Hyper)Graph Neural Networks

## Polynomial Sheaf Diffusion (PolySD)

### Copyright © 2026 Alessio Borgi*, Gabriele Onorato*, Kristjan Tarantelli*, Luke Braithwaite, Fabrizio Silvestri, Pietro Liò

**Polynomial Sheaf Diffusion (PolySD)** is an optimisation of **Neural Sheaf Diffusion**, obtained by replacing fixed spectral filters with **orthogonal-polynomial spectral filters**. This lets you shape diffusion dynamics with families like **Chebyshev (Types I–IV), Legendre, Gegenbauer, and Jacobi**, on both **discrete** and **ODE (continuous-time)** sheaf diffusion variants, and across **Diagonal / Bundle / General** sheaf maps.

This optimisation allows to reach new **state-of-the-art accuracy performances** in **both Homophilic** and **Heterophilic Benchmarks**, and using way **less number of parameters** with higher performances with respect to Neural Sheaf Diffusion, even with NSD having more layers or hidden channels.

---

## 1. Installation

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable Python dependency management.

### Prerequisites

- Python 3.13
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

### Setup

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   or see [website](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)
   ```
      pip install uv
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/alessioborgi/Heterogeneous-Polynomial-Sheaf-Diffusion.git
   cd Heterogeneous-Polynomial-Sheaf-Diffusion
   ```

3. **Install dependencies**:
   ```bash
   uv sync
   ```

   **macOS users**: If you encounter linking issues, run this before `uv sync`:
   ```bash
   export LDFLAGS="${LDFLAGS/--no-as-needed/}"
   ```

4. **Run scripts**:
   ```bash
   uv run python polynsd/scripts/stuff.py
   ```

5. **Run sweep**:
   ```uv run wandb sweep --project  Heterogeneous_Polynomial_SD sweeps/nc/dblp/diag_sheaf.yaml```
   and then the agent.
### Platform-Specific Notes

The project automatically configures Python 3.13, PyTorch and PyG extensions for your platform:
- **macOS**: CPU-MPS builds
- **Linux**: CUDA 12.8 builds
- **Windows**: CUDA 12.8 builds

**Linux GLIBC note (important for CUDA builds):**
The CUDA PyG wheels used by this project require a newer GLIBC than many HPC clusters provide (e.g., GLIBC 2.32+).
If you see errors like `GLIBC_2.32 not found`, run inside a container with a newer base image.

**Container option (recommended if your host GLIBC is too old):**
1. Build the image:
   ```bash
   podman build -t hetero-polysd:latest .
   ```
2. Run a shell with GPU access (adjust volume paths for your system):
   ```bash
   podman run -it --rm \
     -v /home/$USER/Heterogeneous-Polynomial-Sheaf-Diffusion:/work/project \
     -v /mnt/ssd2/$USER/hetero:/work/data \
     --device nvidia.com/gpu=all \
     --ipc host \
     hetero-polysd:latest \
     /bin/bash
   ```
3. Inside the container, launch sweeps as usual:
   ```bash
   uv run wandb sweep --project Heterogeneous_Polynomial_SD sweeps/nc/dblp/diag_sheaf.yaml
   uv run wandb agent sheaf_hypergraphs/Heterogeneous_Polynomial_SD/<SWEEP_ID>
   ```

### Development Dependencies

Install additional development tools:
```bash
uv sync --group dev
```

This includes `pytest` and `pytest-cov` for testing.

---

## 2. Polynomial Filters

PolySD applies a polynomial filter of degree `K` to a rescaled Laplacian operator. Indeed, every Polynomial Sheaf Diffusion layer, gets in input the sheaf laplacian (in its normalised or non normalised version), together with `lambda-max`, which is the upper bound of the spectrum. Its value depends on the typology of sheaf laplacian:
- If `sheaf-laplacian=normalised`: The lambda max is set as:  `lambda-max = 2`, since the sepctrum is bounded in `[0,2]`.
- If `sheaf-laplacian=unnormalised`: The lambda max gets the value depending on the choice we we here: 
    - `lambda_max_choice=analytic` Uses a known bound (e.g., **2** for normalized Laplacians) or closed-form where available. It is based on the Gershgorin's Theorem. 
    - `lambda_max_choice=iterative` Estimates \(\lambda_{\max}\) via power iteration, being a safer solution for **non-standard / sheaf Laplacians**.

**Supported Orthogonal Families**

| `polynomial_type`  | Symbol                   | Interval | Constraints                         |
|--------------------|--------------------------|----------|-------------------------------------|
| `ChebyshevType1`   | \(T_k\)                  | \([-1,1]\) | –                                   |
| `ChebyshevType2`   | \(U_k\)                  | \([-1,1]\) | –                                   |
| `ChebyshevType3`   | \(V_k\)                  | \([-1,1]\) | –                                   |
| `ChebyshevType4`   | \(W_k\)                  | \([-1,1]\) | –                                   |
| `Legendre`         | \(P_k\)                  | \([-1,1]\) | –                                   |
| `Gegenbauer`       | \(C_k^{(\lambda)}\)      | \([-1,1]\) | `gegenbauer_lambda > 0`             |
| `Jacobi`           | \(P_k^{(\alpha,\beta)}\) | \([-1,1]\) | `jacobi_alpha > -1`, `jacobi_beta > -1` |

**Practical Tips**
- Begin with `K ∈ {4, 8, 12}`; higher `K` increases capacity **and** cost.
- Prefer `iterative` lambda for **General** sheaf Laplacians or custom operators.
- Gegenbauer/Jacobi add response-shape control — scan a few values (e.g., λ ∈ {0.5, 1.0, 1.5}).
- In `homophilic settings` prefer `smaller K`, while for `heterophilic settings` prefer `larger K`. 


---

## 3. Citing
For citing the **Polynomial Sheaf Diffusion** paper:

```
@misc{polysd2025,
  title={Polynomial Sheaf Diffusion},
  author={Alessio Borgi and collaborators},
  year={2025},
  note={Code: https://github.com/<user>/Polynomial-Sheaf-Diffusion}
}
```

This repository is based in part on the **Neural Sheaf Diffusion** paper:

```
@inproceedings{bodnar2022neural,
  title={Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing in {GNN}s},
  author={Cristian Bodnar and Francesco Di Giovanni and Benjamin Paul Chamberlain and Pietro Li{\`o} and Michael M. Bronstein},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022},
  url={https://openreview.net/forum?id=vbPsD-BhOZ}
}
```
## 4. License

See `LICENSE` for details.
