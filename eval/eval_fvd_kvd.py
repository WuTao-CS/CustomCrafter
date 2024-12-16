import argparse, os, sys, json, glob
sys.path.insert(0, os.getcwd())
import numpy as np
from utils.cal_fvd_text2video import cal_fvd
import torch
from tqdm import tqdm
from extralibs.pytorch_i3d import InceptionI3d

def get_filelist(data_dir):
    file_list = glob.glob(os.path.join(data_dir, '*.mp4'))
    file_list.sort()
    return file_list

def calc_FVD(real_dir, fake_dir):
    real_list = get_filelist(real_dir)
    fake_list = get_filelist(fake_dir)
    assert len(real_list) == len(fake_list), "Error: samples are not paired for real and fake folders!"
    n_samples = len(real_list)
    real_set = []
    fake_set = []
    from decord import VideoReader, cpu
    for idx in tqdm(range(n_samples), desc='Data Loading'):
        real_reader = VideoReader(real_list[idx], ctx=cpu(0))
        fake_reader = VideoReader(fake_list[idx], ctx=cpu(0))
        real_frames = real_reader.get_batch(list(range(len(real_reader))))
        fake_frames = fake_reader.get_batch(list(range(len(fake_reader)))) # [t,h,w,c]
        real_frames = real_frames.asnumpy().astype(np.uint8)
        fake_frames = fake_frames.asnumpy().astype(np.uint8)
        real_set.append(real_frames)
        fake_set.append(fake_frames)
    real = np.stack(real_set, axis=0)
    fake = np.stack(fake_set, axis=0) # [b,t,h,w,c]
    print(f'finish data loading')
    del VideoReader, cpu
    fvd, kvd, n_samples = cal_fvd(real, fake, device=torch.device("cuda"), batch_size=1, n_sample=n_samples, i3d_path=args.i3d_path)
    return fvd, kvd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", type=str, default=None, help="generated videos saving dir")
    parser.add_argument("--fake_dir", type=str, default=None, help="real videos saving dir")
    parser.add_argument("--save_dir", type=str, default=None, help="results saving dir")
    parser.add_argument("--i3d_path", type=str, default="/apdcephfs_cq2/share_1290939/yingqinghe/dependencies/tats/i3d_pretrained_400.pt", help="pretrained i3d ckpt")
    args = parser.parse_args()
    
    fvd, kvd = calc_FVD(args.real_dir, args.fake_dir)
    print("fvd: %lf; kvd: %lf"%(fvd, kvd))

    # save res
    res = {'FVD': float(fvd),
           'KVD': float(kvd)
           }
    f = open(os.path.join(args.save_dir, f'results_fvd_kvd.json'), 'w')
    json.dump(res, f)
    f.close()