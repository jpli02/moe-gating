import torch

class InclusiveCumsumOp(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, dim: int) -> torch.Tensor:
        if len(x.size()) == 1:
            x = x.view([1, -1])
            out = torch.empty_like(x)
            # ops.inclusive_cumsum(x, 1, out)
            
            return out.squeeze()
        out = torch.empty_like(x)
        # ops.inclusive_cumsum(x, dim, out)
        
        return out


inclusive_cumsum = InclusiveCumsumOp.apply
