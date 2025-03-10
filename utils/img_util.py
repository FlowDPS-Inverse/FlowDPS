from typing import Optional, Union

import numpy as np
import torch
from torch.fft import fft2, fftshift, ifft2, ifftshift
from torchvision.utils import save_image


def draw_img(img: Union[torch.Tensor, np.ndarray],
            save_path:Optional[str]='test.png',
            nrow:Optional[int]=8,
            normalize:Optional[bool]=True):
    if isinstance(img, np.ndarray):
        img = torch.Tensor(img)

    save_image(img, fp=save_path, nrow=nrow, normalize=normalize)

def normalize(img: Union[torch.Tensor, np.ndarray]) \
                        -> Union[torch.Tensor, np.ndarray]:
    
    return (img - img.min())/(img.max()-img.min())
     
def to_np(img: torch.Tensor,
          mode: Optional[str]='NCHW') -> np.ndarray:

    assert mode in ['NCHW', 'NHWC']
    
    if mode == 'NCHW':
        img = img.permute(0,2,3,1) 

    return img.detach().cpu().numpy()

def fft2d(img: torch.Tensor,
          mode: Optional[str]='NCHW') -> torch.Tensor:

    assert mode in ['NCHW', 'NHWC']
    
    if mode == 'NCHW':
        return fftshift(fft2(img))
    elif mode == 'NHWC':
        img = img.permute(0,3,1,2)
        return fftshift(fft2(img))
    else:
        raise NameError    
    

def ifft2d(img: torch.Tensor,
           mode: Optional[str]='NCHW') -> torch.Tensor:

    assert mode in ['NCHW', 'NHWC']
    
    if mode == 'NCHW':
        return ifft2(ifftshift(img))
    elif mode == 'NHWC':
        img = ifft2(ifftshift(img))
        return img.permute(0,2,3,1)
    else:
        raise NameError    


"""
Helper functions for new types of inverse problems
"""

def fft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.fft``.
    Returns:
        The FFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.fftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data


def ifft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.
    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.ifft``.
    Returns:
        The IFFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data

def fft2(x):
  """ FFT with shifting DC to the center of the image"""
  return torch.fft.fftshift(torch.fft.fft2(x), dim=[-1, -2])


def ifft2(x):
  """ IFFT with shifting DC to the corner of the image prior to transform"""
  return torch.fft.ifft2(torch.fft.ifftshift(x, dim=[-1, -2]))


def fft2_m(x):
  """ FFT for multi-coil """
  if not torch.is_complex(x):
      x = x.type(torch.complex64)
  return torch.view_as_complex(fft2c_new(torch.view_as_real(x)))


def ifft2_m(x):
  """ IFFT for multi-coil """
  if not torch.is_complex(x):
      x = x.type(torch.complex64)
  return torch.view_as_complex(ifft2c_new(torch.view_as_real(x)))