# From wikipedia. MATLAB code,
# function x = conjgrad(A, b, x)
#     r = b - A * x;
#     p = r;
#     rsold = r' * r;
# 
#     for i = 1:length(b)
#         Ap = A * p;
#         alpha = rsold / (p' * Ap);
#         x = x + alpha * p;
#         r = r - alpha * Ap;
#         rsnew = r' * r;
#         if sqrt(rsnew) < 1e-10
#             break
#         end
#         p = r + (rsnew / rsold) * p;
#         rsold = rsnew;
#     end
# end

from typing import Callable, Optional

import torch


def CG(A: Callable,
       b: torch.Tensor,
       x: torch.Tensor,
       m: Optional[int]=5,
       eps: Optional[float]=1e-4,
       damping: float=0.0,
       use_mm: bool=False) -> torch.Tensor:
    
    if use_mm:
        mm_fn = lambda x, y: torch.mm(x.view(1, -1), y.view(1, -1).T)
    else:
        mm_fn = lambda x, y: (x * y).flatten().sum()
    
    orig_shape = x.shape
    x = x.view(x.shape[0], -1)

    r = b - A(x)
    p = r.clone()

    rsold = mm_fn(r, r)
    assert not (rsold != rsold).any(), print(f'NaN detected 1')

    for i in range(m):
        Ap = A(p) + damping * p
        alpha = rsold / mm_fn(p, Ap)
        assert not (alpha != alpha).any(), print(f'NaN detected 2')
        
        x = x + alpha * p
        r = r - alpha * Ap

        rsnew = mm_fn(r, r)
        assert not (rsnew != rsnew).any(), print('NaN detected 3')
        
        if rsnew.sqrt().abs() < eps:
            break

        p = r + (rsnew / rsold) * p
        rsold = rsnew.clone()

    return x.reshape(orig_shape)

