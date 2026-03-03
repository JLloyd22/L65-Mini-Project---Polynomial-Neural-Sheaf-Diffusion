#!/bin/sh

python -m exp.run \
    --dataset=ieee24 \
    --d=3 \
    --layers=4 \
    --hidden_channels=20 \
    --left_weights=True \
    --right_weights=True \
    --lr=0.02 \
    --maps_lr=0.005 \
    --input_dropout=0.0 \
    --dropout=0.7 \
    --use_act=True \
    --model=DiagSheafPolynomial \
    --polynomial_type="ChebyshevType1" \
    --normalised=True \
    --deg_normalised=False \
    --sparse_learner=True \
    --lambda_max_choice="iterative" \
    --chebyshev_layers_K=15 \
    --early_stopping=200 \
    --weight_decay=0.005 \
    --folds=1 \
    --epochs=2 \
    --cuda=-1 \
    --entity="jelloyd22-university-of-cambridge" \
    --wandb_project="powergrids_graph"
