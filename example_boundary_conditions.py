#%% Imports
from utils import ker_matrix
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd


#%% Define the interval, the target functions, and the test points
a, b = 0, 1

f = {'1': lambda x, alpha: x ** alpha + x ** 2,
     '2': lambda x, alpha: x ** alpha + alpha * (1 - alpha) / 6 * x ** 3,
     '3': lambda x, alpha: x ** alpha + np.polyval([(-alpha + 5 * alpha ** 2 - 2 * alpha ** 3) / 52, 
                                                    (8 * alpha - 14 * alpha ** 2 + 3 * alpha ** 3) / 39, 
                                                    0, 
                                                    (16 * alpha - 28 * alpha ** 2 + 6 * alpha ** 3) / 13,
                                                    0], x)
     }


x_test = np.linspace(a, b, 10001)


#%% Define the grid of values of alpha and n (number of points)
alphas = np.linspace(3/2, 6, 20, dtype=np.float128)
nn = np.unique(np.logspace(np.log10(500), np.log10(1000), 20).astype('i'))


#%% Compute the interpolants and store the errors

# Lists to store results and later build a dataframe
Npts, Hfill, Alpha, Fun, Metric, Error = [], [], [], [], [], []

# Run over all number of points
for n in tqdm(nn):
    # Define equally space poins, and remove the boundary
    x = np.linspace(a, b, 2 * n+1)[1::2] 
    # Compute the fill distance
    h = np.diff(x)[0]
    # Construct the kernel matix
    A = ker_matrix(x, x, a, b)
    # Construct the evaluation matrix
    A_eval = ker_matrix(x_test, x, a, b)
    # Run over each value of alpha
    for alpha in alphas:
        # Run over each target function
        for fun in f:
            # Evaluate the test data
            y_test = f[fun](x_test, alpha)
            # Evaluate the train data
            y_train = f[fun](x, alpha)
            # Solve the linear system
            c = np.linalg.solve(A, y_train)
            # Compute the pointwise test error
            diff = np.abs(y_test - A_eval @ c)
           
            # Compute and store the inf error
            Npts += [n]
            Hfill += [h]
            Alpha += [alpha]
            Fun += [fun]
            Metric += ['Einf']
            Error += [np.max(diff)]

            # Compute and store the 2 error
            Npts += [n]
            Hfill += [h]
            Alpha += [alpha]
            Fun += [fun]
            Metric += ['E2']
            Error += [np.sqrt(np.mean(diff ** 2))]


#%% Store the results in a dataframe
results = pd.DataFrame({'n': Npts, 'h': Hfill, 'alpha': Alpha, 'fun': Fun, 'metric': Metric, 'error': Error})

            
#%% Compute the rates
# Initialize empty dicts
rates = {'Einf': {}, 'E2': {}}
# Run over the two metrics
for metric in rates:
    # Run over the target functions
    for fun in f:
        # Define a list to store rates
        rates[metric][fun] = []
        # Run over each value of alpha
        for alpha in alphas:
            # Select the values corresponding to (fun, metric, alpha)
            select = results.loc[(results.fun==fun) & (results.metric==metric) & (results.alpha==alpha)]
            # Extract the number of points
            n = select.n
            # Extract the values of the error
            error = select.error
            # Compute the rates as the first coeff of a linear fit of log(n) vs log(error)
            rates[metric][fun] += [-np.polyfit(np.log(n), np.log(error), 1)[0]]
            # rates[metric][fun] += [avg_lin_fit(np.log(n), np.log(error))]

        
#%% Plot the rates
aa = [2.5, 3.5, 4.5]

for idx, metric in enumerate(rates):
    fig = plt.figure(idx)
    fig.clf()
    ax = fig.gca()
    for fun in f:
        ax.plot(alphas, rates[metric][fun], '.-')
    for a in aa:
        if metric == 'Einf':
            a -= 1/2
        ax.hlines(a, np.min(alphas), np.max(alphas), 
                  color='k', linestyle='--', alpha=0.5)
    ax.grid(True)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('rate')
    ax.legend([r'$f_1$', r'$f_2$', r'$f_3$'])
    ax.set_title(metric)

                   