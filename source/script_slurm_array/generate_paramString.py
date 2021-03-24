import numpy as np
from scipy.stats import loguniform
import sys

def generate_parameters(seed):
    np.random.seed(seed)
    out={}
    out['nfeatures'] = np.random.randint(3, 25)
    out['lr'] = float(loguniform.rvs(0.001, 0.01, size=1))
    out['gamma'] = np.random.uniform(0.75, 0.05)
    out['penalty'] = float(loguniform.rvs(0.00001, 0.1, size=1))
    out['batch'] = np.random.choice([32,64])
    return out

if __name__ == '__main__':
    out = generate_parameters(int(sys.argv[1]))
    out_str = '--nfeatures {} --lr {} --gamma {} --penalty {} --batch {}'.format(out['nfeatures'], out['lr'], out['gamma'], out['penalty'], out['batch'])
    print(out_str)
