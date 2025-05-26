%%%% Superconvergence example with periodic kernels

clear all; close all; clc;

addpath('aux/')
rng(42)

% Basic parameters
plot_during_run = false;

% Define kernel
r = 2;
bernoulli_numbers = bernoulli(0:2*r);
bernoulli_coefs = flip(flip(bernoulli_numbers) ./ factorial(0:2*r) ./ factorial(2*r:-1:0) );
bernoulli_poly = @(x) polyval(bernoulli_coefs, x);
ck = (-1)^(r+1) * (2*pi)^(2*r);
k = @(d) 1 + ck * bernoulli_poly(d);

% Dense grid
nxs = 1e5 + 1;
xs = linspace(0, 1, nxs)';

% Parameters for data-generating functions
n_trunc = 1e3; % Truncation point of the KL expansion
js = 1:n_trunc; 

% General parameters
ns = 100:10:200; % Numbers of points for computing the rates
n_samples = 100; % Number of sample functions
rfs = linspace(0.5, 2.5*r, (2.5*r-0.5)*10 + 1);
n_rfs = length(rfs);
rates = zeros(n_rfs, 3);
f_samples = zeros(nxs, n_rfs, n_samples);

% Pre-compute things
n_num = length(ns);
cs = cell(n_num, 1);
Xs_cell = cell(n_num, 1);
for ind_n = 1:n_num
    n = ns(ind_n);
    Xs = linspace(1/n, 1-1/n, n)';    
    c = k(squareform(pdist(Xs))) \ k(pdist2(Xs, xs));
    cs{ind_n} = c;
    Xs_cell{ind_n} = Xs;
end

cos_evals = cos(2*pi*js.*xs);

% Main loops
for i = 1:n_rfs
    rf = rfs(i); % The function will now be in all spaces of smoothness r < rf
    jsrf = js.^(-rf-0.5);
    cos_evals_jsrf = jsrf .* cos_evals;   
    ps = zeros(n_samples, 3);
    for j = 1:n_samples
        % Define data-generating function via Fourier series        
        coefs = randi(3, 1, n_trunc) - 2;
        fxs = 1 + sum( coefs(1, :) .* cos_evals_jsrf, 2);
        f_samples(:, i, j) = fxs;
        
        % Compute errors
        errs = [];
        for ind_n = 1:n_num
            n = ns(ind_n);            
            Xs = Xs_cell{ind_n};
            Ys = interp1(xs, fxs, Xs);
            sxs = cs{ind_n}' * Ys;
            err1 = sum(abs(fxs - sxs).^1) / nxs;
            err2 = sqrt( sum((fxs - sxs).^2) / nxs );    
            err_inf = max(abs(fxs - sxs));
            errs = [errs; err1, err2, err_inf];
        end        
        % Extract and save rates
        p1 = polyfit(log(ns), log(errs(:,1)), 1);
        p2 = polyfit(log(ns), log(errs(:,2)), 1);
        p_inf = polyfit(log(ns), log(errs(:,3)), 1);
        ps(j, :) = [abs(p1(1)), abs(p2(1)), abs(p_inf(1))];
        % Print progress
        [i, j; n_rfs, n_samples]
        
        % Some plotting
        if plot_during_run
            subplot(2, 1, 1)
            loglog(ns, errs, ns, ns(1)^(-rf)*errs(1)*ns.^(-rf), '--', 'LineWidth', 2)
            legend('err L1', 'err L2', 'err Linf', 'polynomial rate')

            subplot(2, 1, 2)
            plot(xs, fxs, xs, sxs, 'LineWidth', 2)
            legend('f', 's')
            pause 
        end
    end
    rates(i, :) = mean(ps, 1);    
end

% Auxiliary functions
function sxs = interpolant(k, X, Y, xs)
    % Assumes that each row represents one point
    K = k(squareform(pdist(X)));
    kXxs = k(pdist2(X, xs));
    sxs = kXxs' * (K \ Y);
end 

%% Plotting
plot(rfs, rates)
legend('err L1', 'err L2', 'err Linf')