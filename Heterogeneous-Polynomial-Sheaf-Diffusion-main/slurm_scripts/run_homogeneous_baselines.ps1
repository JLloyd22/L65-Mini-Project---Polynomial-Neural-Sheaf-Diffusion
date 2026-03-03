# Run all homogeneous GNN baselines on node classification

Write-Host "Running GCN..." -ForegroundColor Green
uv run python polynsd/scripts/run_gnn_nc.py model=gcn dataset.homogeneous=true

Write-Host "Running GAT..." -ForegroundColor Green
uv run python polynsd/scripts/run_gnn_nc.py model=gat dataset.homogeneous=true

Write-Host "Running GIN..." -ForegroundColor Green
uv run python polynsd/scripts/run_gnn_nc.py model=gin dataset.homogeneous=true

Write-Host "Running GraphSAGE..." -ForegroundColor Green
uv run python polynsd/scripts/run_gnn_nc.py model=sage dataset.homogeneous=true

Write-Host "All homogeneous baseline runs completed!" -ForegroundColor Cyan
