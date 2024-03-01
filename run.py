import torch
from torch import nn
torch.set_printoptions(10)
import sys, os

DIR="./" #change absolute path here
sys.path.append(DIR+"src/")

from N3LO import N3L0
from Models import FFNN
from Writers import WriteToFile

from utils import gaussLegendre, wf_train
from utils import braPsi_ketPsi, calc_overlap, calc_energy
from utils import sync_time

import argparse

parser = argparse.ArgumentParser(description='Machine Learning the Deuteron')

parser.add_argument("-H", "--hidden_nodes", type=int,  default=10,       help="Number of hidden neurons per layer")
parser.add_argument("--preepochs",          type=int,  default=10_000,   help="Number of pre-epochs for the pretraining phase")
parser.add_argument("--epochs",             type=int,  default=250_000,  help="Number of epochs for the energy minimisation phase")

args = parser.parse_args()

N=64
b=1.5
alpha=1e-4

hidden_nodes=args.hidden_nodes
act_func=nn.Softplus()
preepochs=args.preepochs
epochs=args.epochs

kmin=0
kmax=500

dtype=torch.float32
device=torch.device('cpu') #small network, CPU is faster!
dtype_str=str(torch.get_default_dtype()).split('.')[-1]

net = FFNN(input_nodes=1,
           hidden_nodes=hidden_nodes,
           output_nodes=2,
           act_func=act_func,
           N=N)
net=net.to(dtype=dtype,device=device)

n3lo = N3L0(filename=DIR+'src/2d_vNN_J1.dat',
            filename2=DIR+'src/wfk.dat',
            dtype=dtype,
            device=device)

optim = torch.optim.Adam(params=net.parameters(), lr=alpha)

model_path_pt = DIR+"results/pretrain/checkpoints/H%03i_%s_P%06i_%s_PT_%s_device_%s_dtype_%s_chkp.pt" % \
                (hidden_nodes, act_func.__class__.__name__, preepochs, optim.__class__.__name__, True, device, dtype_str)
filename_pt = DIR+"results/pretrain/data/H%03i_%s_P%06i_%s_PT_%s_device_%s_dtype_%s.csv" % \
                (hidden_nodes, act_func.__class__.__name__, preepochs, optim.__class__.__name__, True, device, dtype_str)


if(os.path.isfile(filename_pt)):
    load=filename_pt 
else:
    load=None
writer_pt = WriteToFile(load=load,filename=filename_pt)

start_preepoch=0
if(os.path.isfile(model_path_pt)):
    state_dict = torch.load(f=model_path_pt, map_location=device)
    net.load_state_dict(state_dict['model_state_dict'])
    optim.load_state_dict(state_dict['optim_state_dict'])
    start_preepoch=state_dict['epoch']

k, w = gaussLegendre(N=N,kmin=kmin,kmax=kmax,dtype=dtype,device=device)

psi_s_train = wf_train(b=b, k=k)
psi_d_train = k.pow(2) * wf_train(b=b, k=k)

wf_s_train = psi_s_train/torch.sqrt(braPsi_ketPsi(weights=w, kvalues=k, wavef1=psi_s_train, wavef2=psi_s_train))
wf_d_train = psi_d_train/torch.sqrt(braPsi_ketPsi(weights=w, kvalues=k, wavef1=psi_d_train, wavef2=psi_d_train))

for epoch in range(start_preepoch,preepochs):
    stats = {}

    t0=sync_time(device=device)

    psi_s, psi_d = net(k)

    A_s = braPsi_ketPsi(weights=w, kvalues=k, wavef1=psi_s, wavef2=psi_s)
    A_d = braPsi_ketPsi(weights=w, kvalues=k, wavef1=psi_d, wavef2=psi_d)

    wf_s = psi_s/torch.sqrt(A_s)
    wf_d = psi_d/torch.sqrt(A_d)

    K_s = calc_overlap(weights=w, kvalues=k, wavef1=wf_s, wavef2=wf_s_train)
    K_d = calc_overlap(weights=w, kvalues=k, wavef1=wf_d, wavef2=wf_d_train)
    K_loss = (K_s-1)**2 + (K_d-1)**2

    optim.zero_grad()
    K_loss.backward()
    optim.step()

    t1=sync_time(device=device)

    stats['epoch'] = epoch
    stats['Ks'] = K_s.item()
    stats['Kd'] = K_d.item()
    stats['K_loss'] = K_loss.item()
    stats['walltime'] = t1-t0

    writer_pt(stats)

    if(epoch%1000==0):
        sys.stdout.write(f"Epoch: {epoch:6d} Ks: {K_s:4.2e} As: {A_s:4.2e} Kd: {K_d:4.2e} Ad: {A_d:4.2e} Walltime: {(t1-t0):4.2e}s       \r")
        sys.stdout.flush()
        
        torch.save({'epoch':epoch,
                    'model_state_dict':net.state_dict(),
                    'optim_state_dict':optim.state_dict(),
                    'Ks':K_s.item(),
                    'Kd':K_d.item(),
                    'loss':K_loss.item()},
                    model_path_pt)
        writer_pt.write_to_file(filename_pt)
        
sys.stdout.write("\nPretraining DONE! \n")

optim = torch.optim.Adam(params=net.parameters(), lr=alpha)

model_path = DIR+"results/energy/checkpoints/H%03i_%s_P%06i_%s_PT_%s_device_%s_dtype_%s_chkp.pt" % \
                (hidden_nodes, act_func.__class__.__name__, preepochs, optim.__class__.__name__, False, device, dtype_str)
filename = DIR+"results/energy/data/H%03i_%s_P%06i_%s_PT_%s_device_%s_dtype_%s.csv" % \
                (hidden_nodes, act_func.__class__.__name__, preepochs, optim.__class__.__name__, False, device, dtype_str)
if(os.path.isfile(filename)):
    load=filename
else:
    None
writer = WriteToFile(load=load,filename=filename)

start_epoch=0
if(os.path.isfile(model_path)):
    state_dict = torch.load(f=model_path, map_location=device)
    net.load_state_dict(state_dict['model_state_dict'])
    optim.load_state_dict(state_dict['optim_state_dict'])
    start_epoch=state_dict['epoch']

vNN_S, vNN_D, vNN_SD, vNN_DS = n3lo.getOrbitalPotentials()
kk, gs_wfs, gs_wfd = n3lo(k)
deut_energy = calc_energy(weights=w, kvalues=k, kpvalues=k, wavef1=gs_wfs, wavef2=gs_wfd,
                          vNN_11=vNN_S, vNN_22=vNN_D, vNN_12=vNN_SD, vNN_21=vNN_DS)

print(f"Deuteron: {deut_energy:.6f} MeV")
for epoch in range(start_epoch,epochs):
    stats = {}

    t0=sync_time(device=device)

    psi_s, psi_d = net(k)

    A_s = braPsi_ketPsi(weights=w, kvalues=k, wavef1=psi_s, wavef2=psi_s)
    A_d = braPsi_ketPsi(weights=w, kvalues=k, wavef1=psi_d, wavef2=psi_d)

    psi_s = psi_s / torch.sqrt(A_s + A_d)
    psi_d = psi_d / torch.sqrt(A_s + A_d)

    E = calc_energy(weights=w, kvalues=k, kpvalues=k, wavef1=psi_s, wavef2=psi_d,
                    vNN_11=vNN_S, vNN_22=vNN_D, vNN_12=vNN_SD, vNN_21=vNN_DS)
    
    optim.zero_grad()
    E.backward()
    optim.step()

    F_s = calc_overlap(weights=w, kvalues=k, wavef1=psi_s, wavef2=gs_wfs)
    F_d = calc_overlap(weights=w, kvalues=k, wavef1=psi_d, wavef2=gs_wfd)

    t1 = sync_time(device=device)

    stats['epoch'] = epoch
    stats['Fs'] = F_s.item()
    stats['Fd'] = F_d.item()
    stats['E'] = E.item()
    stats['exact'] = deut_energy.item()
    stats['walltime'] = t1-t0

    writer(stats)

    if(epoch%1000==0):
        sys.stdout.write(f"Epoch: {epoch:6d} E: {E:2.6f} MeV Fs: {F_s:2.6f} Fd: {F_d:2.6f} Walltime {(t1-t0):4.2e}s          \r")
        sys.stdout.flush()
        
        torch.save({'epoch':epoch,
                    'model_state_dict':net.state_dict(),
                    'optim_state_dict':optim.state_dict(),
                    'Fs':F_s.item(),
                    'Fd':F_d.item(),
                    'energy':E.item(),
                    'exact':deut_energy.item()},
                    model_path)
        writer.write_to_file(filename)


sys.stdout.write("\nTraining DONE! \n")