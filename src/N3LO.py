import torch
from torch import nn, Tensor

import numpy as np

from constants import hbar

from scipy.special import roots_legendre
from scipy import interpolate

class N3L0(nn.Module):

    def __init__(self, filename: str, filename2: str, dtype: torch.dtype, device: torch.device) -> None:
        super(N3L0, self).__init__()
        
        N=64
        self.dtype=dtype
        self.device=device

        self.k = torch.zeros((N,N), dtype=dtype, device=device)
        self.kp = torch.zeros((N,N), dtype=dtype, device=device)

        self.vNN_S = torch.zeros((N,N), dtype=dtype, device=device)
        self.vNN_D = torch.zeros((N,N), dtype=dtype, device=device)
        self.vNN_SD = torch.zeros((N,N), dtype=dtype, device=device)
        self.vNN_DS = torch.zeros((N,N), dtype=dtype, device=device)

        f = np.genfromtxt(filename)

        for ik in range(N):
            for jk in range(N):
                self.k[ik,jk], self.kp[ik,jk], _, _, self.vNN_S[ik,jk], self.vNN_D[ik,jk], self.vNN_SD[ik,jk], self.vNN_DS[ik,jk] = f[64*ik+jk,:]

        self.V = torch.zeros((2*N,2*N), dtype=dtype, device=device)

        for ik in range(N):
            for jk in range(N):
                self.V[ik,jk] = self.vNN_S[ik,jk]*(hbar**3)
                self.V[ik+N,jk] = self.vNN_DS[ik,jk]*(hbar**3)
                self.V[ik,jk+N] = self.vNN_SD[ik,jk]*(hbar**3)
                self.V[ik+N,jk+N] = self.vNN_D[ik,jk]*(hbar**3)

        out = np.genfromtxt(filename2)
        ksd=out[...,0]
        wfs=out[...,1]
        wfd=out[...,2]

        self.ksd = torch.as_tensor(ksd, dtype=dtype, device=device)#.unsqueeze(-1)
        self.wfs = torch.as_tensor(wfs, dtype=dtype, device=device)#.unsqueeze(-1)
        self.wfd = torch.as_tensor(wfd, dtype=dtype, device=device)#.unsqueeze(-1)

    def getOrbitalPotentials(self):
        return self.vNN_S*(hbar**3),self.vNN_D*(hbar**3),self.vNN_SD*(hbar**3),self.vNN_DS*(hbar**3)

    def forward(self, input_range):
        _k = self.ksd.cpu().detach().numpy()
        _wfs = self.wfs.cpu().detach().numpy()
        _wfd = self.wfd.cpu().detach().numpy()

        f_s = interpolate.interp1d(_k, _wfs, fill_value='extrapolate', kind='linear')
        f_d = interpolate.interp1d(_k, _wfd, fill_value='extrapolate', kind='linear')

        gs_k=input_range.cpu().detach().numpy().flatten()

        gs_wfs = f_s(gs_k)
        gs_wfd = f_d(gs_k)

        gs_k = torch.as_tensor(gs_k, dtype=self.dtype, device=self.device).unsqueeze(1)
        gs_wfs = torch.as_tensor(gs_wfs, dtype=self.dtype, device=self.device).unsqueeze(1)
        gs_wfd = torch.as_tensor(gs_wfd, dtype=self.dtype, device=self.device).unsqueeze(1)
        return gs_k, gs_wfs, gs_wfd