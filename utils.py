import subprocess
import numpy as np
import torch

def rand_idx(n:int, test_ratio=0.2):
    a = np.arange(n)
    np.random.shuffle(a)
    num_train = int(n*(1-test_ratio))
    return a[:num_train], a[num_train:]

class Items(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        assert(X.shape[0] == Y.shape[0])
        
        self.length  = X.shape[0]
        self.X = X
        self.Y = Y
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, i):
        xx = torch.from_numpy(self.X[i])
        yy = torch.from_numpy(np.asarray(self.Y[i]))
        return xx, yy
        
class MultipleOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

class MultipleScheduler(object):
    def __init__(self, *sc):
        self.scheduler = sc

    def step(self):
        for sc in self.scheduler:
            sc.step()

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

DEFAULT_ATTRIBUTES = (
    'index',
    'memory.free',
    'memory.total'
)

def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=DEFAULT_ATTRIBUTES, no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [ line.strip() for line in lines if line.strip() != '' ]
    return [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]

def find_gpu():
    free_gpu = '0'
    free_memory = 0
    for gpu in get_gpu_info():
        if int(gpu['memory.total']) - int(gpu['memory.free']) < 100:
            return 'cuda:' + gpu['index']
        
        if int(gpu['memory.free']) > free_memory:
            free_memory = int(gpu['memory.free'])
            free_gpu = gpu['index']

    return 'cuda:'+free_gpu

def make_data_loader(X, y, batch_size):
    train_index, test_index = rand_idx(X.shape[0], test_ratio = 0.2)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    train_loader = torch.utils.data.DataLoader(Items(X_train,y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(Items(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader