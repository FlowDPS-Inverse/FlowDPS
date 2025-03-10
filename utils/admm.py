import torch


def solve_for_k(x, y, v, u, rho):
    Fx = torch.fft.fftn(x, dim=(-2, -1))
    Fxc = torch.conj(Fx)
    FxcFx = torch.pow(torch.abs(Fx), 2)
    Fy = torch.fft.fftn(y, dim=(-2, -1))

    Sec = torch.fft.fftn((v - u/rho), dim=(-2, -1))
    Fk_new = (Fxc * Fy + rho * Sec) / (FxcFx + rho)

    return torch.real(torch.fft.ifftn(Fk_new, dim=(-2, -1)))

def solve_for_k_v2(x, y, v, u, rho):
    Fx = torch.fft.fftn(x, dim=(-2, -1))
    Fxc = torch.conj(Fx)
    FxcFx = torch.pow(torch.abs(Fx), 2)
    Fy = torch.fft.fftn(y, dim=(-2, -1))
    Sec = torch.fft.fftn((v - u/rho), dim=(-2, -1))

    FR = Fxc * Fy + rho * Sec
    FBR = Fx.mul(FR)
    invWBR = FBR.div(FxcFx + rho)
    FCBInvWBR = Fxc * invWBR
    Fk = (FR - FCBInvWBR) / rho
    return torch.real(torch.fft.ifftn(Fk, dim=(-2, -1)))

def estimate_ker(x, y, k, m:int=15, rho=5e-2, beta=1e-3):
    # initialize 
    v = torch.zeros_like(x).to('cuda')
    u = torch.zeros_like(x).to('cuda')

    for _ in range(m):
        v = torch.clamp((k + u/rho).abs() - beta/rho, 0) * torch.sign(k + u/rho)
        k = solve_for_k_v2(x, y, v, u, rho)
        u = u + 1.5*rho*(k - v)
    
    return torch.fft.ifftshift(k).flip(dims=(-2, -1))