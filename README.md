# Polynomial Neural Sheaf Diffusion for Power Flow Prediction

Code for *"Polynomial Neural Sheaf Diffusion for the Prediction of Power Flow Across Power Grids"* — Geometric Deep Learning (L65), University of Cambridge.

## Data

This repository does not include the PowerGraph datasets due to file size constraints. To download and set up the data:

1. Go to the [PowerGraph figshare page](https://figshare.com/articles/dataset/PowerGraph/22820534)
2. Download the `dataset_pf_opf` folder (this contains the power flow datasets used in our experiments)
3. Place the contents within the `datasets/` directory at the project root

Your directory structure should look like:

```
PolySheafNeuralNetworks-PowerGrids/
├── datasets/
│   ├── ieee24_node/...
│   ├── ieee39_node/...   
│   ├── ieee118_node/... 
│   ├── uk_node/...  
│   └── ...
└── ...
```

The dataset includes four power grids (IEEE24, IEEE39, IEEE118, UK), each with 34,944 graph snapshots for the node-level power flow task. For more details on the dataset structure, see [Varbella et al. (2024)](https://arxiv.org/abs/2402.02827) and the [PowerGraph GitHub](https://github.com/PowerGraph-Datasets).


## Acknowledgements

This project builds upon the [PolyNSD codebase]((https://github.com/alessioborgi/PolySheafNeuralNetworks-PowerGrids/tree/main)) by Alessio Borgi. We thank Alessio for providing the initial experimental framework.
