#!/bin/sh

/home/jel90/.conda/envs/polysheaf/bin/python -m exp.run \
    --dataset=uk_node \
    --task=regression \
    --d=3 \
    --layers=10 \
    --hidden_channels=40 \
    --left_weights=True \
    --right_weights=True \
    --lr=0.0001 \
    --maps_lr=0.0001 \
    --input_dropout=0.0 \
    --dropout=0.2 \
    --use_act=True \
    --model=DiagSheaf \
    --normalised=True \
    --deg_normalised=False \
    --early_stopping=1000 \
    --weight_decay=0.005 \
    --folds=1 \
    --epochs=100 \
    --sparse_learner=False \
    --edge_feat_dim=0 \
    --snapshot_train_samples_per_epoch=1000 \
    --snapshot_eval_samples=1000 \
    --batch_size=32 \
    --cuda=0 \
    --entity="jelloyd22-university-of-cambridge" \
    --wandb_project="powergrids_node" \
    --seed=2020
    #--polynomial_type="ChebyshevType1" \
    #--lambda_max_choice="iterative" \
    #--chebyshev_layers_K=15 \
