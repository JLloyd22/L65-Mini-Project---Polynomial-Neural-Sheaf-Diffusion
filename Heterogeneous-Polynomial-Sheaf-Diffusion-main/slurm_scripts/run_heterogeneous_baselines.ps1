# Run all heterogeneous GNN baselines on node classification

Write-Host "Running HAN..." -ForegroundColor Green
uv run python polynsd/scripts/run_gnn_nc.py model=han dataset.homogeneous=false

Write-Host "Running HGT..." -ForegroundColor Green
uv run python polynsd/scripts/run_gnn_nc.py model=hgt dataset.homogeneous=false

Write-Host "Running RGCN..." -ForegroundColor Green
uv run python polynsd/scripts/run_gnn_nc.py model=rgcn dataset.homogeneous=false

Write-Host "Running HeteroGNN..." -ForegroundColor Green
uv run python polynsd/scripts/run_gnn_nc.py model=hetero_gcn dataset.homogeneous=false

Write-Host "Running HeteroGAT..." -ForegroundColor Green
uv run python polynsd/scripts/run_gnn_nc.py model=hetero_gat dataset.homogeneous=false

Write-Host "Running HeteroGIN..." -ForegroundColor Green
uv run python polynsd/scripts/run_gnn_nc.py model=hetero_gin dataset.homogeneous=false

Write-Host "Running HeteroGraphSAGE..." -ForegroundColor Green
uv run python polynsd/scripts/run_gnn_nc.py model=hetero_sage dataset.homogeneous=false

Write-Host "All heterogeneous baseline runs completed!" -ForegroundColor Cyan
