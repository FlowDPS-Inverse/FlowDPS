# -*- coding: utf-8 -*-
import torch.fft
import torch

import numpy as np
from scipy import ndimage
from scipy.interpolate import interp2d

def splits(a, sf):
    '''split a into sfxsf distinct blocks
    Args:
        a: NxCxWxH
        sf: split factor
    Returns:
        b: NxCx(W/sf)x(H/sf)x(sf^2)
    '''
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=4)
    b = torch.cat(torch.chunk(b, sf, dim=3), dim=4)
    return b


def p2o(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.
    Args:
        psf: NxCxhxw
        shape: [H, W]
    Returns:
        otf: NxCxHxWx2
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[...,:psf.shape[2],:psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    otf = torch.fft.fftn(otf, dim=(-2,-1))
    #n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    #otf[..., 1][torch.abs(otf[..., 1]) < n_ops*2.22e-16] = torch.tensor(0).type_as(psf)
    return otf


def upsample(x, sf=3):
    '''s-fold upsampler
    Upsampling the spatial size by filling the new entries with zeros
    x: tensor image, NxCxWxH
    '''
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2]*sf, x.shape[3]*sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z


def downsample(x, sf=3):
    '''s-fold downsampler
    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others
    x: tensor image, NxCxWxH
    '''
    st = 0
    return x[..., st::sf, st::sf]


def data_solution_simple(x, F2K, FKFy, rho):
    rho = rho.clip(min=1e-2)
    numerator = FKFy + torch.fft.fftn(rho*x, dim=(-2,-1))
    denominator = F2K + rho
    FX = numerator / denominator
    Xest = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))
    return Xest


def data_solution_nonuniform(x, FK, FKC, F2KM, FKFMy, rho):
    rho = rho.clip(min=1e-2)
    numerator = FKFMy + torch.fft.fftn(rho*x, dim=(-2,-1))
    denominator = F2KM + rho
    FX = numerator / denominator
    Xest = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))
    return Xest


def pre_calculate(x, k):
    '''
    Args:
        x: NxCxHxW, LR input
        k: NxCxhxw

    Returns:
        FK, FKC, F2K
        will be reused during iterations
    '''
    w, h = x.shape[-2:]
    FK = p2o(k, (w, h))
    FKC = torch.conj(FK)
    F2K = torch.pow(torch.abs(FK), 2)
    return FK, FKC, F2K


def pre_calculate_FK(k):
    '''
    Args:
        k: [25, 1, 33, 33] 25 is the number of filters

    Returns:
        FK:
        FKC:
    '''
    # [25, 1, 512, 512] (expanded from) [25, 1, 33, 33]
    FK = p2o(k, (512, 512))
    FKC = torch.conj(FK)
    return FK, FKC


def pre_calculate_nonuniform(x, y, FK, FKC, mask):
    '''
    Args:
        x: [1, 3, 512, 512]
        y: [1, 3, 512, 512]
        FK: [25, 1, 512, 512] 25 is the number of filters
        FKC: [25, 1, 512, 512]
        m: [1, 25, 512, 512]

    Returns:
    '''
    mask = mask.transpose(0, 1)
    w, h = x.shape[-2:]
    # [1, 3, 512, 512] -> [25, 3, 512, 512]
    By = y.repeat(mask.shape[0], 1, 1, 1)
    # [25, 3, 512, 512]
    My = mask * By
    # or use just fft..?
    FMy = torch.fft.fft2(My)
    
    # [25, 3, 512, 512]
    FKFMy = FK * FMy
    # [1, 3, 512, 512]
    FKFMy = torch.sum(FKFMy, dim=0, keepdim=True)
    
    # [25, 1, 512, 512]
    F2KM = torch.abs(FKC * (mask ** 2) * FK)
    # [1, 1, 512, 512]
    F2KM = torch.sum(F2KM, dim=0, keepdim=True)
    return F2KM, FKFMy


def classical_degradation(x, k, sf=3):
    ''' blur + downsampling

    Args:
        x: HxWxC image, [0, 1]/[0, 255]
        k: hxw, double
        sf: down-scale factor

    Return:
        downsampled LR image
    '''
    x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')
    #x = filters.correlate(x, np.expand_dims(np.flip(k), axis=2))
    st = 0
    return x[st::sf, st::sf, ...]



def shift_pixel(x, sf, upper_left=True):
    """shift pixel for super-resolution with different scale factors
    Args:
        x: WxHxC or WxH, image or kernel
        sf: scale factor
        upper_left: shift direction
    """
    h, w = x.shape[:2]
    shift = (sf-1)*0.5
    xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
    if upper_left:
        x1 = xv + shift
        y1 = yv + shift
    else:
        x1 = xv - shift
        y1 = yv - shift

    x1 = np.clip(x1, 0, w-1)
    y1 = np.clip(y1, 0, h-1)

    if x.ndim == 2:
        x = interp2d(xv, yv, x)(x1, y1)
    if x.ndim == 3:
        for i in range(x.shape[-1]):
            x[:, :, i] = interp2d(xv, yv, x[:, :, i])(x1, y1)

    return x
