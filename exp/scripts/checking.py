import torch
labels = torch.load('/mnt/c/Users/jello/OneDrive/Work Stuff/Cambridge MLMI Notes and Work/GDL/PolySheafNeuralNetworks-PowerGrids/labels_snapshot_fold0.pt')
print("Shape:", labels.shape)
print("p_gen — min:", labels[:,0].min().item(), "max:", labels[:,0].max().item())
print("q_gen — min:", labels[:,1].min().item(), "max:", labels[:,1].max().item())
print("v     — min:", labels[:,2].min().item(), "max:", labels[:,2].max().item())
print("theta — min:", labels[:,3].min().item(), "max:", labels[:,3].max().item())