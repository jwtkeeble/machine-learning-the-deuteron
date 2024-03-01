import torch
from torch import Tensor, nn

from scipy.special import roots_legendre
from scipy import interpolate

from constants import hbar, mu

from time import perf_counter

def sync_time(device: torch.device) -> float:
    if(device.type=='cuda'):
        torch.cuda.synchronize()
    return perf_counter()

def gaussLegendre(N, kmin, kmax, dtype, device):
    x, w = roots_legendre(n=N, mu=False)

    x = torch.as_tensor(x, dtype=dtype, device=device)
    w = torch.as_tensor(w, dtype=dtype, device=device)

    k = 0.5*(x+1.0)
    w = 0.5*w

    c = kmax/(torch.tan((torch.pi/2.0)*k[-1]))
    w = (w*0.5*c*torch.pi)/(torch.cos((torch.pi/2.0)*k)**2.0)
    k = c*torch.tan((torch.pi/2.0)*k)

    k = k.unsqueeze(1)
    w = w.unsqueeze(1)

    return k, w

def wf_train(b: float, k: Tensor) -> Tensor:
	return torch.exp(-0.5*(b**2.0)*(k**2.0))

def braPsi_ketPsi(weights: Tensor, kvalues: Tensor, wavef1: Tensor, wavef2: Tensor) -> Tensor:
    return 4.*torch.pi*torch.sum(weights*kvalues.pow(2)*wavef1*wavef2)

def calc_overlap(weights: Tensor, kvalues: Tensor, wavef1: Tensor, wavef2: Tensor) -> Tensor:
    ans12 = braPsi_ketPsi(weights=weights, kvalues=kvalues, wavef1=wavef1, wavef2=wavef2)
    ans11 = braPsi_ketPsi(weights=weights, kvalues=kvalues, wavef1=wavef1, wavef2=wavef1)
    ans22 = braPsi_ketPsi(weights=weights, kvalues=kvalues, wavef1=wavef2, wavef2=wavef2)
    overlap = (ans12**2)/(ans11*ans22)
    return overlap #(overlap-1)**2

def calc_kinetic_energy(weights: Tensor, kvalues: Tensor, wavef: Tensor, kpvalues: Tensor, wavefp: Tensor):
    return 4.0*torch.pi*torch.sum( weights*kvalues.pow(2)*wavefp.mul(wavef)*((kpvalues.mul(hbar)).pow(2)/(2.0*mu)))

def calc_potential_energy(weights: Tensor, kvalues: Tensor, kpvalues: Tensor, wavef: Tensor, wavefp: Tensor, vNN: Tensor):
    term1 = weights*kvalues.pow(2)*wavefp
    temp = weights*kpvalues.pow(2)*wavef
    term2 = torch.matmul(temp.transpose(1,0), vNN).transpose(1,0)
    Ep = (4.0*torch.pi)*torch.sum(term1*term2)
    return Ep

def calc_energy(weights: Tensor, kvalues: Tensor, kpvalues: Tensor, wavef1: Tensor, wavef2: Tensor, vNN_11: Tensor, vNN_22: Tensor, vNN_12: Tensor, vNN_21: Tensor):
    Ek1 = calc_kinetic_energy(weights=weights, kvalues=kvalues, kpvalues=kpvalues, wavef=wavef1, wavefp=wavef1)
    Ek2 = calc_kinetic_energy(weights=weights, kvalues=kvalues, kpvalues=kpvalues, wavef=wavef2, wavefp=wavef2)

    Ep11 = calc_potential_energy(weights=weights, kvalues=kvalues, kpvalues=kpvalues, wavef=wavef1, wavefp=wavef1, vNN=vNN_11)
    Ep12 = calc_potential_energy(weights=weights, kvalues=kvalues, kpvalues=kpvalues, wavef=wavef1, wavefp=wavef2, vNN=vNN_12)
    Ep21 = calc_potential_energy(weights=weights, kvalues=kvalues, kpvalues=kpvalues, wavef=wavef2, wavefp=wavef1, vNN=vNN_21)
    Ep22 = calc_potential_energy(weights=weights, kvalues=kvalues, kpvalues=kpvalues, wavef=wavef2, wavefp=wavef2, vNN=vNN_22)

    A1 = braPsi_ketPsi(weights=weights, kvalues=kvalues, wavef1=wavef1, wavef2=wavef1)
    A2 = braPsi_ketPsi(weights=weights, kvalues=kvalues, wavef1=wavef2, wavef2=wavef2)

    E = (Ek1+Ek2+Ep11+Ep12+Ep21+Ep22)/(A1+A2)
    return E


