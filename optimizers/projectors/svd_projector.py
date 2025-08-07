import torch
import torch.nn.functional as F
from typing import Optional


class SVDProjector:
    """
    SVD-based projector for subspace momentum optimization.
    """
    
    def __init__(self, rank: int, update_gap: int = 1):
        self.rank = rank
        self.update_gap = update_gap
        self.U = None
        self.S = None
        self.V = None
        self.step_count = 0
        
    def project(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Project gradient using SVD basis.
        
        Args:
            grad: Input gradient tensor
            
        Returns:
            Projected gradient tensor
        """
        if self.U is None:
            return grad
        
        # Reshape gradient to 2D for SVD
        grad_2d = grad.view(grad.shape[0], -1)
        
        # Project using U basis
        projected = grad_2d @ self.U @ self.U.T
        
        return projected.view_as(grad)
    
    def project_back(self, proj_grad: torch.Tensor) -> torch.Tensor:
        """
        Project back from subspace to full space.
        
        Args:
            proj_grad: Projected gradient tensor
            
        Returns:
            Full space gradient tensor
        """
        if self.U is None:
            return proj_grad
        
        # Reshape to 2D for projection
        proj_grad_2d = proj_grad.view(proj_grad.shape[0], -1)
        
        # Project back using U basis
        full_grad = proj_grad_2d @ self.U @ self.U.T
        
        return full_grad.view_as(proj_grad)
    
    def update_basis(self, grad: torch.Tensor):
        """
        Update the SVD basis using the current gradient.
        
        Args:
            grad: Current gradient tensor
        """
        self.step_count += 1
        
        if self.step_count % self.update_gap != 0:
            return
        
        # Reshape gradient to 2D for SVD
        grad_2d = grad.view(grad.shape[0], -1)
        
        # Compute SVD
        U, S, V = torch.svd(grad_2d)
        
        # Keep only top-k singular vectors
        if self.rank < U.shape[1]:
            self.U = U[:, :self.rank]
            self.S = S[:self.rank]
            self.V = V[:, :self.rank]
        else:
            self.U = U
            self.S = S
            self.V = V


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