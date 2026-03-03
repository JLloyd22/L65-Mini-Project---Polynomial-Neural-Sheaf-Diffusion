import torch
from pathlib import Path

fold_dir = Path("/mnt/c/Users/jello/OneDrive/Work Stuff/Cambridge MLMI Notes and Work/GDL/PolySheafNeuralNetworks-PowerGrids/plots/23850682_fold0")
combined = {}
for pt_file in sorted(fold_dir.glob("*.pt")):
    layer_name = pt_file.stem  # e.g. "layer_0"
    combined[layer_name] = torch.load(pt_file, map_location="cpu")

torch.save(combined, fold_dir / "restriction_maps.pt")
print("Saved combined restriction_maps.pt")
