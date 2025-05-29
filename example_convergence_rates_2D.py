

import numpy as np
from datetime import datetime
from vkoga.kernels import Matern, Wendland
from matplotlib import pyplot as plt
from utils import compute_conv_rates
import os



# Fixed parameters
dim = 2
f_func = lambda x, alpha: x[:, [0]]**alpha * x[:, [1]]**alpha


exponent_Xfine = 9                     # 10


# Variable parameters
para_ep = 1
k_mat = 0
name_kernel = 'mat{}'.format(k_mat)


# Setup up everything
dic_approx = {'mat0': [Matern(k=0, ep=para_ep), 1.5, list(np.round(np.arange(.1, 3, .1), 2))],
              'mat1': [Matern(k=1, ep=para_ep), 2.5, list(np.round(np.arange(.1, 5, .1), 2))],
              'wendl1': [Wendland(k=1, d=dim, ep=para_ep), 2.5, [.3, .8, 1.3, 1.8, 2.3, 2.8, 3.3, 3.8, 4.3, 4.8, 5.3]],
              'mat2': [Matern(k=2, ep=para_ep), 3.5, list(np.round(np.arange(.1, 7, .1), 2))],}

list_npoints_per_dim = np.unique(np.geomspace(5, 120, 20, dtype=int))

X1 = np.linspace(0, 1, 2**exponent_Xfine + 1).reshape(-1, 1)
X2 = np.linspace(0, 1, 2**exponent_Xfine + 1).reshape(-1, 1)
X, Y = np.meshgrid(X1, X2)
X_fine = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)

array_errors_L2 = np.zeros((len(dic_approx[name_kernel][2]), len(list_npoints_per_dim)))
array_errors_Linfty = np.zeros((len(dic_approx[name_kernel][2]), len(list_npoints_per_dim)))
array_errors_L1 = np.zeros((len(dic_approx[name_kernel][2]), len(list_npoints_per_dim)))

# Run the computation
for idx_alpha, alpha in enumerate(dic_approx[name_kernel][2]):

    print(datetime.now().strftime("%H:%M:%S"), ': Computation {}/{}.'.format(
        idx_alpha + 1, len(dic_approx[name_kernel][2])))

    y_fine = f_func(X_fine, alpha)

    for idx_npoints, npoints_per_dim in enumerate(list_npoints_per_dim):
        print(datetime.now().strftime("%H:%M:%S"), npoints_per_dim)

        # Set up centers
        X = np.linspace(0, 1, npoints_per_dim).reshape(-1, 1)
        X = X[1:-1, :]        # Do not use points on the boundary
        X, Y = np.meshgrid(X, X)
        ctrs = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)

        # Compute interpolant
        coeffs = np.linalg.solve(dic_approx[name_kernel][0].eval(ctrs, ctrs), f_func(ctrs, alpha))

        # Evaluate interpolant (batch-wise!)
        y_pred = np.zeros((X_fine.shape[0], 1))

        size_batch = 1000           # batches of size 1000
        n_batches = y_pred.shape[0] // size_batch + 1
        for idx_batch in range(n_batches):
            y_pred[idx_batch * size_batch : (idx_batch + 1) * size_batch] = \
                dic_approx[name_kernel][0].eval(X_fine[idx_batch * size_batch : (idx_batch + 1) * size_batch, :], ctrs) @ coeffs

        # Compute error
        error_L2_squared = np.mean((y_pred - y_fine) ** 2)**(1/2)
        array_errors_L2[idx_alpha, idx_npoints] = error_L2_squared

        error_Linfty = np.max(np.abs(y_pred - y_fine))
        array_errors_Linfty[idx_alpha, idx_npoints] = error_Linfty

        error_L1 = np.mean(np.abs(y_pred - y_fine))
        array_errors_L1[idx_alpha, idx_npoints] = error_L1





# # Visualization of the convergence rates
tau = (dim + (2*k_mat + 1)) / 2

list_start = [-5, -4]
list_stop = [-2, -1]

plt.figure(11+k_mat)
plt.clf()
list_rates_L1, _ = compute_conv_rates(array_errors_L1, (np.array(list_npoints_per_dim, dtype=float)-1)**(-1), list_start, list_stop)
list_rates_L2, _ = compute_conv_rates(array_errors_L2, (np.array(list_npoints_per_dim, dtype=float)-1)**(-1), list_start, list_stop)
list_rates_Linfty, _ = compute_conv_rates(array_errors_Linfty, (np.array(list_npoints_per_dim, dtype=float)-1)**(-1), list_start, list_stop)
plt.plot(0.5 + np.array(dic_approx[name_kernel][2]), list_rates_L1, 'b*', markersize=5)
plt.plot(0.5 + np.array(dic_approx[name_kernel][2]), list_rates_L2, 'b.', markersize=3)
plt.plot(0.5 + np.array(dic_approx[name_kernel][2]), list_rates_Linfty, 'bx', markersize=5)
plt.plot([0, tau, 2*tau], [0, tau, 2*tau], 'k--x', markersize=10, color='gray')
plt.plot([0, 2*tau], [tau + 1/2, tau + 1/2], 'k--', color='gray')
plt.legend(['L1 error', 'L2 error', 'Linfty error'])
plt.xlabel('sigma')
plt.ylabel('conv rate in fill dist')
plt.title('dim = {}, '.format(dim) + name_kernel)
plt.show(block=False)


