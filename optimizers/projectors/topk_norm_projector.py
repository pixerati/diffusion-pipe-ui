import torch


class TopKNormProjector:
    """
    Top-K norm-based projector for subspace momentum optimization.
    """
    
    def __init__(self, rank: int, update_gap: int = 1):
        self.rank = rank
        self.update_gap = update_gap
        self.indices = None
        self.step_count = 0
        
    def project(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Project gradient using top-K norm selection.
        
        Args:
            grad: Input gradient tensor
            
        Returns:
            Projected gradient tensor
        """
        if self.indices is None:
            return grad
        
        # Reshape gradient to 2D
        grad_2d = grad.view(grad.shape[0], -1)
        
        # Select top-K components
        projected = grad_2d[:, self.indices]
        
        return projected.view(grad.shape[0], self.rank)
    
    def project_back(self, proj_grad: torch.Tensor) -> torch.Tensor:
        """
        Project back from subspace to full space.
        
        Args:
            proj_grad: Projected gradient tensor
            
        Returns:
            Full space gradient tensor
        """
        if self.indices is None:
            return proj_grad
        
        # Reshape to 2D
        proj_grad_2d = proj_grad.view(proj_grad.shape[0], -1)
        
        # Create full tensor and fill in selected components
        grad_2d = torch.zeros(proj_grad_2d.shape[0], self.indices.max() + 1, 
                             device=proj_grad.device, dtype=proj_grad.dtype)
        grad_2d[:, self.indices] = proj_grad_2d
        
        return grad_2d.view_as(proj_grad)
    
    def update_basis(self, grad: torch.Tensor):
        """
        Update the top-K norm basis.
        
        Args:
            grad: Current gradient tensor
        """
        self.step_count += 1
        
        if self.step_count % self.update_gap != 0:
            return
        
        # Reshape gradient to 2D
        grad_2d = grad.view(grad.shape[0], -1)
        
        # Compute norms and select top-K
        norms = torch.norm(grad_2d, dim=0)
        _, indices = torch.topk(norms, min(self.rank, norms.shape[0]))
        
        self.indices = indices 