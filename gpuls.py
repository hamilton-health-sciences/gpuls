import time
import sys
import io
import os
import gc
from functools import reduce

DATA_FILE_ERROR_CODE = 1
SAMPLE_SIZE_ERROR_CODE = 2
DEPENDENCY_ERROR_CODE = 4

def _message(*s):
    print('[MESSAGE]', *s)
    sys.stdout.flush()

def _warning(*s):
    print('[WARNING]', *s)
    sys.stdout.flush()

def _error(*s):
    print(' [ERROR] ', *s)
    sys.stdout.flush()

try:
    import numpy as np
    import scipy as sp
    import scipy.stats
    import scipy.sparse.linalg
    import torch
    from pynvml import *
    import rpy2.rinterface_lib
    import rpy2.robjects as robjects
except:
    _error('Dependencies are not installed! Will try to install now...')
    os.system('pip3 install --user numpy scipy torch pynvml rpy2')
    _message('If installation appears succesful, try running `gpuls` again.')
    sys.exit(DEPENDENCY_ERROR_CODE)

def initialize_all_gpus():
    '''Initialize all GPUs.

    Initialization is performed by coping a single tensor to each visible GPU.
    '''
    torch.cuda.empty_cache()
    for i in range(torch.cuda.device_count()):
        torch.tensor(0).cuda(device = 'cuda:' + str(i))
    nvmlInit()

def number_gpus():
    '''Return the number of available GPUs.'''
    return torch.cuda.device_count()

def available_gpu_memory(i):
    '''Return the available memory on a given GPU in bytes.'''
    handle = nvmlDeviceGetHandleByIndex(i)
    return nvmlDeviceGetMemoryInfo(handle).free

def expected_size(*dims):
    '''Given a tensor t, return the number of bytes expected to be used when
    stored on the GPU as a 64-bit float tensor.
    '''
    num_floats = np.prod(dims)
    return num_floats * 8

def pretty_size(size):
    '''Pretty prints a torch.Size object.'''
    assert(isinstance(size, torch.Size))
    return '  '.join(map(str, size))

def dump_tensors(gpu_only=True):
    '''Prints a list of the Tensors being tracked by the garbage collector.'''
    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if not gpu_only or obj.is_cuda:
                    print('%s:%s%s %s' % (type(obj).__name__, 
                        ' GPU' if obj.is_cuda else '',
                        ' pinned' if obj.is_pinned else '',
                        pretty_size(obj.size())))
                    total_size += obj.numel()
            elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
                if not gpu_only or obj.is_cuda:
                    print('%s  %s:%s%s%s%s %s' % (type(obj).__name__, 
                        type(obj.data).__name__, 
                        ' GPU' if obj.is_cuda else '',
                        ' pinned' if obj.data.is_pinned else '',
                        ' grad' if obj.requires_grad else '', 
                        ' volatile' if obj.volatile else '',
                        pretty_size(obj.data.size())))
                    total_size += obj.data.numel()
        except Exception as e:
            pass        
    _message('Total size:', total_size)

def GxE(G, E, include_E=False, include_GxE=False,
        quantile_G=False, quantile_GxE=False, quantile_E=False):
    '''Concatenate genotype with environmental factors'''
    if quantile_G:
        G = quantile_normalize(G)
    if quantile_E:
        E = quantile_normalize(E)
    if include_GxE:
        GxE = np.multiply(G, E)
        if quantile_GxE:
            GxE = quantile_normalize(GxE)
        X = np.concatenate((G, GxE), axis=1)
    else:
        X = G
    if include_E:
        X = np.concatenate((X, E), axis=1)

    return X

def _quantile_normalize(x):
    xrank = sp.stats.rankdata(x, method='average')
    xqn = sp.stats.norm.ppf(xrank / (len(x) + 1))

    return xqn

def quantile_normalize(X):
    return np.apply_along_axis(_quantile_normalize, 0, X)

def A_cpu(X):
    '''Compute A = X.T . X on the CPU.'''
    if type(X) == np.ndarray:
        X = torch.from_numpy(X)
    return torch.einsum('ij,ik->jk', X, X)

def b_cpu(X, y):
    if type(X) == np.ndarray:
        X = torch.from_numpy(X)
    if type(y) == np.ndarray:
        y = torch.from_numpy(y)
    return torch.einsum('ij,ik->jk', X, y)

def b_gpu_block(X, y):
    '''Compute b = X.T . y on the GPU when X does not fit on the GPU.

    Computation is done by dividing X into blocks by column.
    '''
    _message('Computing b in blocks...')
    padding = X.shape[0] * 1e2
    mem_avail = np.max([available_gpu_memory(i) for i in range(number_gpus())])
    total_tensor_size = mem_avail // 8 - padding
    split = int(total_tensor_size // (2 * X.shape[0] + 1))
    _message('Maximum X block size: {} x {}'.format(X.shape[0], split))
    X_split = torch.split(X, split, dim=1)
    y_gpu = y.cuda()
    b_cpu = []

    for i in range(len(X_split)):
        Xi = X_split[i].cuda()
        bi = torch.matmul(torch.t(Xi), y_gpu).cpu()
        b_cpu.append(bi)

        del Xi
        torch.cuda.empty_cache()

    b = torch.cat(b_cpu, dim=0)

    del X_split, y_gpu, b_cpu
    torch.cuda.empty_cache()
    _message('Done computing b!')

    return b

def beta_gpu(A, b, tol=1e-16):
    '''Run conjugate gradient when A = X.T . X fits on the GPU.

    Returns the vector of coefficients of length P.
    '''
    A_gpu = A.cuda()
    b_gpu = b.cuda()

    x = torch.zeros(b_gpu.size(), dtype=torch.float64).cuda()
    r = b_gpu.clone()
    p = b_gpu.clone()
    rr = torch.sum(r * r)

    while rr / A.shape[0] > tol ** 2:
        Ap = torch.matmul(A_gpu, p)
        alpha = rr / torch.sum(p * Ap)
        x += alpha * p
        r -= alpha * Ap
        rr_new = torch.sum(r * r)
        beta = rr_new / rr
        p = r + beta * p
        rr = rr_new

    x_cpu = x.cpu()

    del A_gpu, b_gpu, x, r, p, Ap, alpha, beta, rr, rr_new
    torch.cuda.empty_cache()

    return x_cpu

def beta_gpu_block(A, b, tol=1e-16):
    '''Run conjugate gradient when A = X.T . X does not fit on the GPU by
    splitting A into blocks.

    Returns the vector of coefficients of length P.
    '''
    padding = A.shape[0] * 5e3
    mem_avail = np.max([available_gpu_memory(i) for i in range(number_gpus())])
    total_tensor_size = mem_avail // 8 - padding
    split = int(total_tensor_size // A.shape[0] - 3)
    A_split = torch.split(A, split, dim=0)
    b_gpu = b.cuda()

    x = torch.zeros(b_gpu.size(), dtype=torch.float64).cuda()
    r = b_gpu.clone()
    p = b_gpu.clone()
    rr = torch.sum(torch.matmul(torch.t(r), r))

    numiter = 0
    while rr / A.shape[0] > tol ** 2:
        numiter += 1
        if numiter % 100 == 0:
            _message('Reached iteration {}'.format(numiter))
        Ap_ = []
        for i in range(len(A_split)):
            Ai = A_split[i].cuda()
            Ap_.append(torch.matmul(Ai, p).cpu())
            del Ai
            torch.cuda.empty_cache()
        Ap = torch.cat(Ap_, dim=0).cuda()

        alpha = rr / torch.sum(torch.matmul(torch.t(p), Ap))
        x += alpha * p
        r -= alpha * Ap
        rr_new = torch.sum(torch.matmul(torch.t(r), r))
        beta = rr_new / rr
        p = r + beta * p
        rr = rr_new
    _message('Converged after {} iterations'.format(numiter))

    x_cpu = x.cpu()

    del A_split, b_gpu, x, r, p, Ap_, Ap, alpha, beta, rr, rr_new
    torch.cuda.empty_cache()

    return x_cpu

def beta_mgpu_block(A, b, tol=1e-16):
    '''Run conjugate gradient on multiple GPUs when A = X.T . X does not fit on
    the GPU by splitting A across GPUs.
    '''
    _message('Computing beta...')
    padding = A.shape[0] * 5e3
    mem_avail = np.max([available_gpu_memory(i) for i in range(number_gpus())])
    total_tensor_size = mem_avail // 8 - padding
    split = int(total_tensor_size // A.shape[0] - 3)
    A_split = torch.split(A, split, dim=0)
    A_ = []
    for i in range(len(A_split)):
        A_.append(A_split[i].cuda(device='cuda:' + str(i)))
    b_gpu = b.cuda(device='cuda:0')

    x = torch.zeros(b_gpu.size(), dtype=torch.float64).cuda(device='cuda:0')
    r = b_gpu.clone().cuda(device='cuda:0')
    p = b_gpu.clone().cuda(device='cuda:0')
    rr = torch.sum(torch.matmul(torch.t(r), r))
    numiter = 0
    while rr / A.shape[0] > tol ** 2:
        numiter += 1
        if numiter % 100 == 0:
            _message('Reached iteration {}'.format(numiter))
        p_, Ap_ = [], []
        for i in range(len(A_)):
            p_.append(p.cuda(device='cuda:' + str(i)))
            Ap_.append(torch.matmul(A_[i], p_[i]).cpu())
        Ap = torch.cat(Ap_, dim=0).cuda(device='cuda:0')
        del p_, Ap_
        torch.cuda.empty_cache()

        alpha = rr / torch.sum(torch.matmul(torch.t(p), Ap))
        x += alpha * p
        r -= alpha * Ap
        rr_new = torch.sum(torch.matmul(torch.t(r), r))
        beta = rr_new / rr
        p = r + beta * p
        rr = rr_new
    _message('Converged after {} iterations'.format(numiter))

    x_cpu = x.cpu()

    del A_, A_split, b_gpu, x, r, p, Ap, alpha, beta, rr, rr_new
    torch.cuda.empty_cache()
    _message('Done computing beta!')

    return x_cpu

def compute_required_gpus(A_shape):
    padding = A_shape[0] * 5e3
    mem_avail = 10_000 * 1_000_000
    total_tensor_size = mem_avail // 8 - padding
    split = int(total_tensor_size // A_shape[0] - 3)

    return int(np.ceil(A_shape[0] / split))

def precond_beta_mgpu_block(A, b, tol=1e-16):
    '''Run conjugate gradient on multiple GPUs when A = X.T . X does not fit on
    the GPU by splitting A across GPUs. Preconditioning is performed using a
    sparse approximate LU factorization with the default options in scipy.
    '''
    _message('Computing beta (using approximate inverse preconditioning of A)...')
    padding = A.shape[0] * 5e3
    mem_avail = np.max([available_gpu_memory(i) for i in range(number_gpus())])
    total_tensor_size = mem_avail // 8 - padding
    split = int(total_tensor_size // A.shape[0] - 3)
    Minv = sp.sparse.linalg.spilu(A.numpy()).solve(np.eye(b.shape[0]))
    Minv = torch.from_numpy(Minv)
    A_split = torch.split(A, split, dim=0)
    A_ = []
    for i in range(len(A_split)):
        A_.append(A_split[i].cuda(device='cuda:' + str(i)))
    Minv_split = torch.split(Minv, split, dim=0)
    Minv_ = []
    for i in range(len(Minv_split)):
        Minv_.append(Minv_split[i].cuda(device='cuda:' + str(i)))
    b_gpu = b.cuda(device='cuda:0')

    x = torch.zeros(b_gpu.size(), dtype=torch.float64).cuda(device='cuda:0')
    r = b_gpu.clone().cuda(device='cuda:0')
    z = torch.matmul(Minv, b).cuda(device='cuda:0')
    p = b_gpu.clone().cuda(device='cuda:0')
    rr = torch.sum(torch.matmul(torch.t(r), r))
    rz = torch.sum(torch.matmul(torch.t(r), z))
    numiter = 0
    while rr > tol ** 2:
        numiter += 1
        if numiter % 100 == 0:
            _message('Reached iteration {}'.format(numiter))

        p_, Ap_ = [], []
        for i in range(len(A_)):
            p_.append(p.cuda(device='cuda:' + str(i)))
            Ap_.append(torch.matmul(A_[i], p_[i]).cpu())
        Ap = torch.cat(Ap_, dim=0).cuda(device='cuda:0')
        del p_, Ap_
        torch.cuda.empty_cache()

        alpha = rz / torch.sum(torch.matmul(torch.t(p), Ap))
        x += alpha * p
        rnew = alpha * Ap
        r_, znew_ = [], []
        for i in range(len(Minv_)):
            r_.append(r.cuda(device='cuda:' + str(i)))
            znew_.append(torch.matmul(Minv_[i], r_[i]).cpu())
        znew = torch.cat(znew_, dim=0).cuda(device='cuda:0')
        beta = torch.sum(znew * (rnew - r)) / rz
        p = znew + beta * p
        r = rnew
        z = znew
        rz = torch.sum(torch.matmul(torch.t(r), z))
    _message('Converged after {} iterations'.format(numiter))

    x_cpu = x.cpu()

    del A_, A_split, b_gpu, x, r, p, Ap, alpha, beta, rr, rr_new
    torch.cuda.empty_cache()
    _message('Done computing beta!')

    return x_cpu

def ypred_gpu_block(X, beta):
    '''Compute predicted values when X does not fit on the GPU. Computations are
    split over the rows of X.

    Returns ypred = X . beta.
    '''
    _message('Computing ypred...')
    padding = X.shape[1] * 5e3
    mem_avail = np.max([available_gpu_memory(i) for i in range(number_gpus())])
    total_tensor_size = mem_avail // 8 - padding
    split = int(total_tensor_size // X.shape[1] - 1)
    X_split = torch.split(X, split, dim=0)
    beta_gpu = beta.cuda()
    ypred = []

    for i in range(len(X_split)):
        Xi = X_split[i].cuda()
        ypredi = torch.matmul(Xi, beta_gpu).cpu()
        ypred.append(ypredi)

        del Xi, ypredi
        torch.cuda.empty_cache()

    ypred = torch.cat(ypred, dim=0)

    del X_split, beta_gpu
    torch.cuda.empty_cache()
    _message('Done computing ypred!')

    return ypred

def R2(y, ypred, intercept=False):
    '''Compute the R^2 coefficient of determination.
    '''
    _message('Computing R^2')

    if intercept:
        ss_tot = torch.sum((y - torch.mean(y, dim=0)) ** 2, dim=0)
    else:
        ss_tot = torch.sum(y ** 2, dim=0)

    ss_res = torch.sum((y - ypred) ** 2, dim=0)
    r2 = 1 - (ss_res / ss_tot)

    return r2

def adj_R2(r2, n, p):
    '''Compute the adjusted R^2 coefficient of determination.
    '''
    _message('Computing adjusted R^2')
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    return adj_r2

def F_test(r2, n, p, intercept=False):
    '''Compute the F-test statistic and p-value.
    '''
    if intercept:
        fac = 0
    else:
        fac = 1
    x = r2 / (1 - r2) * (n - p) / (p - 1 + fac)
    return x, 1 - sp.stats.f.cdf(x, p - 1 + fac, n - p)

def single_gpu_least_squares(X, y):
    '''The entire procedure for when X does not fit on the GPU, but A does.
    '''
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    start = time.time()
    A = A_cpu(X)
    end = time.time()
    _message('Computing A on CPU took: {}'.format(end - start))
    start = time.time()
    b = b_gpu_block(X, y)
    end = time.time()
    _message('Computing b on GPU took: {}'.format(end - start))
    start = time.time()
    beta = beta_gpu_block(A, b)
    end = time.time()
    _message('Computing beta on GPU took: {}'.format(end - start))
    start = time.time()
    ypred = ypred_gpu_block(X, beta)
    r2 = R2(y, ypred)
    adj_r2 = adj_R2(r2, X.shape[0], X.shape[1])
    end = time.time()
    _message('Computing ypred and summary stats took: {}'.format(end - start))

    return beta, ypred, r2, adj_r2

def cpuls(X, y):
    
    '''
    input:
        - X: n x p numpy array
        - y: n x 1 numpy array
    
    output:
        - prediction: n x 1 numpy array
        - beta: p x 1 numpy array (linear regression coefficients)
        - beta p_values: p x 1 numpy array (coefficient t-test p values)
    '''
    A = np.matmul(X.T, X)
    b = np.matmul(X.T, y)
    A_inv = np.linalg.inv(A)
    
    # coefficient
    beta = np.matmul(A_inv.T, b).T
    
    # prediction
    y_pred = np.matmul(X, beta.T)

    # p-value
    MSE = (np.sum((y - y_pred)**2, axis=0))/(X.shape[0] - X.shape[1])
    var_b = np.outer(A_inv.diagonal(), MSE).T
    sd_b = np.sqrt(var_b)
    ts_b = beta / sd_b
    p_values = 2 * (1 - sp.stats.t.cdf(np.abs(ts_b), X.shape[0] - X.shape[1]))
    
    return y_pred, beta, p_values

def load_matrix(fn, k=None):
    try:
        if os.path.exists(fn):
            extension = os.path.splitext(fn)[-1][1:].lower()
            if extension == 'txt':
                try:
                    M = np.loadtxt(fn)
                except:
                    M = np.loadtxt(fn, delimiter=',')
            elif extension == 'h5':
                with h5py.File(fn, 'r') as f:
                    if k is None:
                        k = list(f.keys())[0]
                    M = np.asarray(f[k])
            elif extension == 'rdata':
                rpy2.rinterface_lib.callbacks.consolewrite_warnerror = lambda m: _warning('R error buffer is not empty!')
                x = robjects.r['get'](robjects.r['load'](fn))
                M = np.array(x)
                if len(M.shape) == 2 and M.shape[0] < M.shape[1]:
                    M = M.T
            else:
                raise Exception('"{}" is not a known extension!'.format(extension.lower()))
            _message('{} loaded, shape: {} x {}'.format(fn, *M.shape))
        else:
            raise Exception('"{}" could not be found! Ensure it exists.'.format(fn))
    except Exception as e:
        _error(str(e).split('\n')[0])
        if 'exists' not in str(e):
            _error('Problem loading file "{}", ensure that it is a matrix, not a vector and can be loaded in the R console!'.format(fn))
        sys.exit(DATA_FILE_ERROR_CODE)

    return M

def write_output(out, fn, key=None):
    extension = os.path.splitext(fn)[-1][1:]
    if extension == 'h5':
        with h5py.File(fn, 'w') as f:
            if type(out) == dict:
                for k in out:
                    f[k] = out[k]
            else:
                f[key] = out
    else:
        raise Exception('"{}" is not a known extension!'.format(extension))

def load_X(X_path, E_path, intercept, environment, interaction,
           quantile_g, quantile_gxe, quantile_e):
    X = load_matrix(X_path)
    if E_path:
        E = load_matrix(E_path)
        if X.shape[0] != E.shape[0]:
            _error('X and E do not have the same number of samples!')
            sys.exit(SAMPLE_SIZE_ERROR_CODE)
        X = GxE(X, E, environment, interaction, quantile_g, quantile_gxe, quantile_e)
    if intercept:
        append = np.ones((np.shape(X)[0], 1))
        X = np.concatenate([append, X], axis=1)
    _message('Shape of constructed X: {} x {}'.format(*X.shape))

    return X

if __name__ == '__main__':
    import argparse
    import h5py

    parser = argparse.ArgumentParser(
        description='Fast linear regressions on the GPU using PyTorch.'
    )
    parser.add_argument('-g', '--gpu-guess', type=str, default=None,
                        help='write out the guessed number of GPUs required to do the beta computation')
    parser.add_argument('-i', '--intercept', dest='intercept', action='store_const',
                        const=True, default=False, help='whether to include an intercept term')
    parser.add_argument('-n', '--interaction', dest='interaction', action='store_const',
                        const=True, default=False, help='whether to include a GxE term')
    parser.add_argument('-t', '--environment', dest='environment', action='store_const',
                        const=True, default=False, help='whether to include an E term')
    parser.add_argument('-qx', '--quantile-x', dest='quantile_g', action='store_const',
                        const=True, default=False,
                        help='whether to quantile normalize the G term')
    parser.add_argument('-qi', '--quantile-interaction', dest='quantile_gxe', action='store_const',
                        const=True, default=False,
                        help='whether to quantile normalize the GxE term')
    parser.add_argument('-qe', '--quantile-e', dest='quantile_e', action='store_const',
                        const=True, default=False,
                        help='whether to quantile normalize the E term')
    parser.add_argument('-p', '--precondition', dest='precondition', action='store_const',
                        const=True, default=False,
                        help='whether to precondition A prior to computing beta')
    parser.add_argument('-X', type=str, help='path to X matrix')
    parser.add_argument('-e', type=str, help='path to E matrix')
    parser.add_argument('-y', type=str, help='path to y vector')
    parser.add_argument('-Ab', type=str, help='path to A matrix')
    parser.add_argument('-o', type=str, required=True,
                        help='path to output')
    
    parser.add_argument('--compute-Ab', dest='action', action='store_const',
                        const='Ab', help='compute A = X.T . X and b = X.T . y')
    parser.add_argument('--compute-ls', dest='action', action='store_const',
                        const='ls', help='compute the least-squares solution')
    parser.add_argument('--compute-slow', dest='action', action='store_const',
                        const='slow', help='compute slowly, but with p-values')
    args = parser.parse_args()

    if args.action == 'slow':
        X = load_X(args.X, args.e, args.intercept, args.environment,
                   args.interaction, args.quantile_g, args.quantile_gxe,
                   args.quantile_e)
        y = load_matrix(args.y)
        if X.shape[0] != y.shape[0]:
            _error('X and Y do not have the same number of samples!')
            sys.exit(SAMPLE_SIZE_ERROR_CODE)
        ypred, beta, pvalue = cpuls(X, y)
        N, P = X.shape
        del X
        r2 = R2(torch.from_numpy(y), torch.from_numpy(ypred), args.intercept)
        del y
        adj_r2 = adj_R2(r2, N, P)
        fstat, f_p_value = F_test(r2, N, P, args.intercept)
        write_output({'beta': beta, 'ypred': ypred, 'r2': r2, 'adj_r2': adj_r2,
                      'f_statistic': fstat, 'f_p_value': f_p_value,
                      't_p_values': pvalue},
                     args.o)
    elif args.action == 'Ab':
        X = torch.from_numpy(load_X(args.X, args.e, args.intercept,
                                    args.environment, args.interaction,
                                    args.quantile_g, args.quantile_gxe,
                                    args.quantile_e))
        y = torch.from_numpy(load_matrix(args.y))
        if X.shape[0] != y.shape[0]:
            _error('X and Y do not have the same number of samples!')
            sys.exit(SAMPLE_SIZE_ERROR_CODE)
        A = A_cpu(X)
        b = b_cpu(X, y)
        A /= X.shape[0]
        b /= X.shape[0]
        ngpus_required = compute_required_gpus(A.shape)
        write_output({'A': A, 'b': b}, args.o)
        if args.gpu_guess:
            with open(args.gpu_guess, 'w') as f:
                f.write('{}'.format(ngpus_required))
    elif args.action == 'ls':
        initialize_all_gpus()
        A = torch.from_numpy(load_matrix(args.Ab, 'A'))
        b = torch.from_numpy(load_matrix(args.Ab, 'b'))
        if number_gpus() > 1:
            if args.precondition:
                beta = precond_beta_mgpu_block(A, b)
            else:
                beta = beta_mgpu_block(A, b)
        else:
            beta = beta_gpu_block(A, b)
        del A, b
        X = torch.from_numpy(load_X(args.X, args.e, args.intercept, args.environment, args.interaction,
                                    args.quantile_g, args.quantile_gxe, args.quantile_e))
        y = torch.from_numpy(load_matrix(args.y))
        N, P = X.shape
        ypred = ypred_gpu_block(X, beta)
        del X
        r2 = R2(y, ypred, args.intercept)
        del y
        adj_r2 = adj_R2(r2, N, P)
        fstat, f_p_value = F_test(r2, N, P, args.intercept)
        write_output({'beta': beta, 'ypred': ypred, 'r2': r2, 'adj_r2': adj_r2,
                      'f_statistic': fstat, 'f_p_value': f_p_value},
                     args.o)
    else:
        raise Exception('Unknown action {}'.format(args.action))

