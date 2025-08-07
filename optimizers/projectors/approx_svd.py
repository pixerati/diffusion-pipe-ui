import torch
import torch.nn.functional as F
from typing import Optional


class ApproxSVDProjector:
    """
    Approximate SVD projector for subspace momentum optimization.
    Uses power iteration to approximate the top singular vectors.
    """
    
    def __init__(self, rank: int, update_gap: int = 1, num_iterations: int = 5):
        self.rank = rank
        self.update_gap = update_gap
        self.num_iterations = num_iterations
        self.U = None
        self.S = None
        self.V = None
        self.step_count = 0
        
    def project(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Project gradient using approximate SVD basis.
        
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
        Update the approximate SVD basis using power iteration.
        
        Args:
            grad: Current gradient tensor
        """
        self.step_count += 1
        
        if self.step_count % self.update_gap != 0:
            return
        
        # Reshape gradient to 2D
        grad_2d = grad.view(grad.shape[0], -1)
        
        # Compute approximate SVD using power iteration
        U, S, V = self._approximate_svd(grad_2d)
        
        # Keep only top-k singular vectors
        if self.rank < U.shape[1]:
            self.U = U[:, :self.rank]
            self.S = S[:self.rank]
            self.V = V[:, :self.rank]
        else:
            self.U = U
            self.S = S
            self.V = V
    
    def _approximate_svd(self, matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute approximate SVD using power iteration.
        
        Args:
            matrix: Input matrix
            
        Returns:
            tuple: (U, S, V) approximate SVD components
        """
        m, n = matrix.shape
        k = min(self.rank, m, n)
        
        # Initialize random matrices
        device = matrix.device
        dtype = matrix.dtype
        
        U = torch.randn(m, k, device=device, dtype=dtype)
        V = torch.randn(n, k, device=device, dtype=dtype)
        
        # Power iteration
        for _ in range(self.num_iterations):
            # Update U
            U_new = matrix @ V
            U, _ = torch.linalg.qr(U_new)
            
            # Update V
            V_new = matrix.T @ U
            V, _ = torch.linalg.qr(V_new)
        
        # Compute singular values
        S = torch.diag(U.T @ matrix @ V)
        
        return U, S, V 