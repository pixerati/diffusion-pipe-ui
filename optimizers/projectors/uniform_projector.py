import torch
import torch.nn.functional as F


class UniformProjector:
    """
    Uniform random projection for subspace momentum optimization.
    """
    
    def __init__(self, rank: int, update_gap: int = 1):
        self.rank = rank
        self.update_gap = update_gap
        self.projection_matrix = None
        self.step_count = 0
        
    def project(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Project gradient using uniform random projection.
        
        Args:
            grad: Input gradient tensor
            
        Returns:
            Projected gradient tensor
        """
        if self.projection_matrix is None:
            return grad
        
        # Reshape gradient to 2D for projection
        grad_2d = grad.view(grad.shape[0], -1)
        
        # Project using random matrix
        projected = grad_2d @ self.projection_matrix.T
        
        return projected.view(grad.shape[0], self.rank)
    
    def project_back(self, proj_grad: torch.Tensor) -> torch.Tensor:
        """
        Project back from subspace to full space.
        
        Args:
            proj_grad: Projected gradient tensor
            
        Returns:
            Full space gradient tensor
        """
        if self.projection_matrix is None:
            return proj_grad
        
        # Project back using random matrix
        full_grad = proj_grad @ self.projection_matrix
        
        return full_grad.view_as(proj_grad)
    
    def update_basis(self, grad: torch.Tensor):
        """
        Update the random projection basis.
        
        Args:
            grad: Current gradient tensor
        """
        self.step_count += 1
        
        if self.step_count % self.update_gap != 0:
            return
        
        # Reshape gradient to 2D
        grad_2d = grad.view(grad.shape[0], -1)
        
        # Generate random projection matrix
        device = grad.device
        dtype = grad.dtype
        dim = grad_2d.shape[1]
        
        self.projection_matrix = torch.randn(self.rank, dim, device=device, dtype=dtype)
        self.projection_matrix = F.normalize(self.projection_matrix, dim=1) 