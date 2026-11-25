'''
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import numpy as np
import torch
from torch.utils.data import TensorDataset


def load(name,seed):
    rng=np.random.default_rng(seed)
    def read(z,key,dtype=np.float32):
        with z.open(key) as f:
            return torch.from_numpy(np.loadtxt(f,dtype=dtype))
    with ZipFile(Path(__file__).parent/'datasets'/f'{name}.zip') as outer:
        if name=='mfeat':
            features=(read(outer,'mfeat-fac'),read(outer,'mfeat-pix'),torch.arange(10).repeat_interleave(200))
            indices=np.stack([rng.permutation(200)+i*200 for i in range(10)])
            parts=[]
            for group in np.split(indices,[140,170],axis=1):
                idx=rng.permutation(group.ravel())
                parts.append(tuple(x[idx] for x in features))
        else:
            with ZipFile(BytesIO(outer.read('UCI HAR Dataset.zip'))) as z:
                def partition(split):
                    base=f'UCI HAR Dataset/{split}/'
                    def sensor(key):
                        return torch.cat([read(z,f'{base}Inertial Signals/{key}_{axis}_{split}.txt') for axis in 'xyz'],1)
                    return sensor('total_acc'),sensor('body_gyro'),read(z,base+f'y_{split}.txt',np.int64)-1
                subjects=read(z,'UCI HAR Dataset/train/subject_train.txt',np.int64)
                val=torch.isin(subjects,torch.from_numpy(rng.permutation(subjects.unique().numpy())[:4]))
                train=partition('train')
                parts=[tuple(x[~val] for x in train),tuple(x[val] for x in train),partition('test')]
    mean=[x.mean(0) for x in parts[0][:2]]
    std=[x.std(0,unbiased=False) for x in parts[0][:2]]
    std=[torch.where(s>1e-6,s,1.) for s in std]
    return [TensorDataset((a-mean[0])/std[0],(b-mean[1])/std[1],y) for a,b,y in parts],mean,std
'''

from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset


def load(name, seed):
    rng = np.random.default_rng(seed)
    data_dir = Path(__file__).parent / 'datasets'

    def read(z, key, dtype=np.float32):
        with z.open(key) as f:
            return torch.from_numpy(np.loadtxt(f, dtype=dtype))

    if name == 'cmu-mosi':
        pkl_path = data_dir / f'{name}.pkl'
        if pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                raw = pickle.load(f)
        else:
            with ZipFile(data_dir / f'{name}.zip') as outer:
                pkl_name = next(n for n in outer.namelist() if n.endswith('.pkl'))
                with outer.open(pkl_name) as f:
                    raw = pickle.load(f)

        def partition(split):
            d = raw[split]
            feats = []
            for k in ('text', 'audio', 'vision'):
                x = np.nan_to_num(np.asarray(d[k], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                t = torch.from_numpy(x)
                if t.ndim == 3:
                    t = t.mean(1)
                feats.append(t)
            label_key = 'regression_labels' if 'regression_labels' in d else 'labels'
            y = np.asarray(d[label_key], dtype=np.float32).reshape(-1)
            y = torch.from_numpy((y >= 0).astype(np.int64))
            return (*feats, y)

        val_key = 'valid' if 'valid' in raw else 'val'
        parts = [partition('train'), partition(val_key), partition('test')]
    else:
        with ZipFile(data_dir / f'{name}.zip') as outer:
            if name == 'mfeat':
                features = (read(outer, 'mfeat-fac'), read(outer, 'mfeat-pix'), torch.arange(10).repeat_interleave(200))
                indices = np.stack([rng.permutation(200) + i * 200 for i in range(10)])
                parts = []
                for group in np.split(indices, [140, 170], axis=1):
                    idx = rng.permutation(group.ravel())
                    parts.append(tuple(x[idx] for x in features))
            else:
                with ZipFile(BytesIO(outer.read('UCI HAR Dataset.zip'))) as z:
                    def partition(split):
                        base = f'UCI HAR Dataset/{split}/'
                        def sensor(key):
                            return torch.cat([read(z, f'{base}Inertial Signals/{key}_{axis}_{split}.txt') for axis in 'xyz'], 1)
                        return sensor('total_acc'), sensor('body_gyro'), read(z, base + f'y_{split}.txt', np.int64) - 1
                    subjects = read(z, 'UCI HAR Dataset/train/subject_train.txt', np.int64)
                    val = torch.isin(subjects, torch.from_numpy(rng.permutation(subjects.unique().numpy())[:4]))
                    train = partition('train')
                    parts = [tuple(x[~val] for x in train), tuple(x[val] for x in train), partition('test')]

    mean = [x.mean(0) for x in parts[0][:-1]]
    std = [x.std(0, unbiased=False) for x in parts[0][:-1]]
    std = [torch.where(s > 1e-6, s, 1.) for s in std]
    return [TensorDataset(*[(x - m) / s for x, m, s in zip(xs, mean, std)], y) for *xs, y in parts], mean, std