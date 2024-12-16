import os
import sys
import math
import time
import json
import argparse
import numpy as np
from tqdm import trange
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import torch

from utils.fvd_utils import get_fvd_logits, frechet_distance, load_fvd_model, load_stylegan_v_i3d, polynomial_mmd
from utils.utils import load_npz_from_paths

device = torch.device('cuda')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--run_id', type=int, default=None)
    parser.add_argument('--res_dir', type=str, default=None)
    parser.add_argument('--n_sample', type=int, default=2048)
    args = parser.parse_args()
    return args

def extract_feat_from_np(data,batch_size,i3d,device,progress=True):
    n_batch = math.ceil(data.shape[0] / batch_size)
    embeddings = []
    if progress:
        iterator = trange(n_batch, desc="Extract Features")
    else:
        iterator = range(n_batch)

    for i in iterator:
        embeddings.append(get_fvd_logits(data[i*batch_size:(i+1)*batch_size], i3d=i3d, device=device, batch_size=batch_size))
    embeddings = torch.cat(embeddings, 0)
    return embeddings

def run(args, run_id, i3d, device):
    start = time.time()

    # real_paths = [os.path.join(args.sample_dir, dn) for dn in os.listdir(args.sample_dir) if dn.startswith("real_")]
    # real_paths = [os.path.join(args.sample_dir, dn) for dn in os.listdir(args.sample_dir) if "real" in os.path.join(args.sample_dir, dn)]
    real_paths = []
    for r, dirs, files in os.walk(args.sample_dir):
        for fn in files:
            fp = os.path.join(r, fn)
            print(fp)
            if fp.endswith('.npz') and "real" in fp:
                real_paths.append(fp)
    
    print(args.sample_dir)
    print(real_paths)

    real_data = load_npz_from_paths(real_paths)
    print(f"load real data: {real_data.shape}")
    real_embeddings = extract_feat_from_np(real_data,args.batch_size,i3d,device)[:args.n_sample]
    if real_data.shape[0] < args.n_sample:
        raise ValueError

    # fake_paths = [os.path.join(args.sample_dir, dn) for dn in os.listdir(args.sample_dir) if dn.startswith("fake_")]
    fake_paths = []
    for r, dirs, files in os.walk(args.sample_dir):
        for fn in files:
            fp = os.path.join(r, fn)
            if fp.endswith('.npz') and "fake" in fp:
                fake_paths.append(fp)
    # fake_paths = [os.path.join(args.sample_dir, dn) for dn in os.listdir(args.sample_dir) if "fake" in os.path.join(args.sample_dir, dn)]
    fake_data = load_npz_from_paths(fake_paths)
    print(f"load fake data: {fake_data.shape}")
    if fake_data.shape[0] < args.n_sample:
        raise ValueError
    fake_embeddings = extract_feat_from_np(fake_data,args.batch_size,i3d,device)[:args.n_sample]

    fvd = frechet_distance(fake_embeddings, real_embeddings).cpu().numpy() # np float32
    kvd = polynomial_mmd(fake_embeddings.cpu(), real_embeddings.cpu()) # np float 64

    total = time.time() - start
    print(f'Run_id = {run_id}')
    print(f'FVD = {fvd:.2f}')
    print(f'KVD = {kvd:.2f}')
    print(f'Time = {total:.2f}')
    return [fvd, kvd, total]

def load_data(input):
    if isinstance(input, np.ndarray):
        return input
    elif isinstance(input, str):
        if os.path.isfile(input) and input.endswith('npz'):
            return np.load(input)['arr_0']
        elif os.path.isfile(input) and input.endswith('npy'):
            return np.load(input)
        elif os.path.isdir(input):
            raise NotImplementedError
        else:
            raise ValueError
    else:
        raise ValueError
            
def cal_fvd(real_samples, fake_samples, device, batch_size, n_sample, i3d_path=None, progress=True):
    start = time.time()
    
    # model
    print(f'load_fvd_model ... ')
    i3d = load_fvd_model(device, i3d_path)

    real_data = load_data(real_samples)
    print(f'real data shape = {real_data.shape}')
    if real_data.shape[0] < n_sample:
        print(f'num real samples ({real_data.shape[0]}) is smaller than target n_sample ({n_sample}), change n_sample to {real_data.shape[0]}')
    real_embeddings = extract_feat_from_np(real_data, batch_size, i3d, device,progress)[:n_sample]
    del real_data
    fake_data = load_data(fake_samples)
    print(f'fake data shape = {fake_data.shape}')
    if fake_data.shape[0] < n_sample:
        print(f'num fake samples ({fake_data.shape[0]}) is smaller than target n_sample ({n_sample}), change n_sample to {fake_data.shape[0]}')
    fake_embeddings = extract_feat_from_np(fake_data, batch_size, i3d, device,progress)[:n_sample]
    del fake_data
    fvd = frechet_distance(fake_embeddings, real_embeddings).cpu().numpy() # np float32
    kvd = polynomial_mmd(fake_embeddings.cpu(), real_embeddings.cpu()) # np float 64
    total = time.time() - start
    n_samples = fake_embeddings.shape[0]

    '''
    res={'FVD': f'{fvd:.2f}',
         'KVD': f'{kvd:.2f}',
         'Time': f'{total}',
         'Num_samples': f'{n_samples}',
         }
    f = open(os.path.join(logdir, f'fvd_results.json'), 'w')
    json.dump(res, f)
    f.close()
    '''
    return fvd, kvd, n_samples

def run_multitimes(args, i3d, device):
    res_all = []
    for i in range(args.n_runs):
        run_id = args.run_id if args.run_id is not None else i
        res = run(args, run_id, i3d=i3d, device=device)
        res_all.append(np.array(res))
    res_avg = np.mean(np.stack(res_all, axis=0), axis=0)
    res_std = np.std(np.stack(res_all, axis=0), axis=0)
    
    print(f'{args.n_runs} runs')
    print(f'MEAN: {res_avg}')
    print(f'STD: {res_std}')
    
    print(f'FVD = {res_avg[0]} ({res_std[0]})')
    print(f'KVD = {res_avg[1]} ({res_std[1]})')
    print(f'Time = {res_avg[2]} ({res_std[2]})')
    
    # dump results
    res={'FVD': f'{res_avg[0]} ({res_std[0]})',
         'KVD': f'{res_avg[1]} ({res_std[1]})',
         'Time': f'{res_avg[2]} ({res_std[2]})',
         'Clip_path': f'{args.sample_dir}'
         }
    f = open(os.path.join(args.res_dir, f'{args.n_runs}runs_fvd_stat.json'), 'w')
    json.dump(res, f)
    f.close()

if __name__ == '__main__':
    args = get_args()
    if args.run_id is not None:
        assert(args.n_runs == 1)
    if args.res_dir is None:
        args.res_dir = args.sample_dir
    else:
        os.makedirs(args.res_dir, exist_ok=True)
    
    # load i3d
    i3d = load_fvd_model(device)
    # i3d = load_stylegan_v_i3d(device) # input is 0-255 torch.int8...

    # go
    run_multitimes(args, i3d, device)
    