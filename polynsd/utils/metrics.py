#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

import torch
from torch import Tensor
from torchmetrics import Metric


class MeanReciprocalRank(Metric):
    """Mean Reciprocal Rank (MRR) for link prediction.
    
    Computes MRR by ranking positive edges against all edges in the batch.
    This is a simplified version suitable for batch-wise computation.
    """
    
    full_state_update: bool = False
    higher_is_better: bool = True
    
    def __init__(self):
        super().__init__()
        self.add_state("sum_reciprocal_ranks", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_positives", default=torch.tensor(0.0), dist_reduce_fx="sum")
    
    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update MRR metric.
        
        For each positive edge, finds its rank when all edges are sorted by score.
        Higher scores = better rank.
        
        Args:
            preds: Predicted scores (shape: [batch_size])
            target: Binary labels (1 for positive, 0 for negative)
        """
        if preds.dim() == 2:
            preds = preds.squeeze(-1)
        
        target = target.flatten().to(torch.long)
        preds = preds.flatten()
        
        positive_mask = target == 1
        num_positives = positive_mask.sum()
        
        if num_positives == 0:
            return
        
        # Get predictions for positive samples
        positive_preds = preds[positive_mask]
        
        # Sort all predictions in descending order (higher score = better)
        sorted_preds, _ = torch.sort(preds, descending=True)
        
        # For each positive, find how many predictions are strictly better
        # Using searchsorted: find insertion point in descending sorted array
        # Convert to ascending for searchsorted, then convert rank back
        sorted_preds_asc = sorted_preds.flip(0)
        positive_preds_sorted = positive_preds.sort(descending=False)[0]
        
        # Find positions in ascending sorted array
        positions = torch.searchsorted(sorted_preds_asc, positive_preds_sorted, right=False)
        
        # Convert to ranks in descending order (1-indexed)
        ranks = (len(preds) - positions).float()
        ranks = torch.clamp(ranks, min=1.0)  # Ensure minimum rank is 1
        
        # Calculate reciprocal ranks
        reciprocal_ranks = 1.0 / ranks
        
        self.sum_reciprocal_ranks += reciprocal_ranks.sum()
        self.num_positives += num_positives.float()
    
    def compute(self) -> Tensor:
        """Compute mean reciprocal rank."""
        if self.num_positives == 0:
            return torch.tensor(0.0, device=self.sum_reciprocal_ranks.device)
        return self.sum_reciprocal_ranks / self.num_positives


class HitsAtK(Metric):
    """Hits@K metric for link prediction.
    
    Measures the proportion of positive edges that appear in the top-K predictions.
    """
    
    full_state_update: bool = False
    higher_is_better: bool = True
    
    def __init__(self, k: int = 10):
        super().__init__()
        self.k = k
        self.add_state("hits", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
    
    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update Hits@K metric.
        
        Args:
            preds: Predicted scores (shape: [batch_size])
            target: Binary labels (1 for positive, 0 for negative)
        """
        if preds.dim() == 2:
            preds = preds.squeeze(-1)
        
        target = target.flatten().to(torch.long)
        preds = preds.flatten()
        
        positive_mask = target == 1
        num_positives = positive_mask.sum()
        
        if num_positives == 0:
            return
        
        # Get top-k predictions
        k = min(self.k, len(preds))
        _, top_k_indices = torch.topk(preds, k)
        
        # Check how many positives are in top-k
        top_k_mask = torch.zeros_like(target, dtype=torch.bool)
        top_k_mask[top_k_indices] = True
        
        hits_in_top_k = (positive_mask & top_k_mask).sum()
        
        self.hits += hits_in_top_k.float()
        self.total += num_positives.float()
    
    def compute(self) -> Tensor:
        """Compute Hits@K."""
        if self.total == 0:
            return torch.tensor(0.0, device=self.hits.device)
        return self.hits / self.total
