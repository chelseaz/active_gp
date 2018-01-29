#!/usr/bin/python

import argparse
import numpy as np
import os
import sys

from scipy.linalg import cholesky, solve_triangular

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from viz import *

fig_prefix = "figs/"

def filename_for(strategy, ground_truth, N_init, N_final, custom):
    return fig_prefix + "%s_%s_%s_%d_%d.png" % \
        (ground_truth, strategy, custom, N_init, N_final)


class CovariateSpace():
    def __init__(self, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax

    def sample(self, n, rng):
        # sample x uniformly for now
        # eventually incorporate density of x
        return rng.uniform(self.xmin, self.xmax, n)


class GroundTruth():
    # mean_fn takes in a numpy array and returns a scalar output
    # noise_fn takes in a random state and returns a scalar output
    def __init__(self, variance, mean_fn, noise_fn, name):
        self.variance = variance
        self.mean_fn = mean_fn
        self.noise_fn = noise_fn
        self.name = name
        self.observe_y_fn = lambda x, rng: mean_fn(x) + noise_fn(rng)

        # TODO: calculate this programmatically
        # hardcoded for the low_freq_sinusoid based on sequential version with 1000 pts
        self.approx_length_scale = 0.334


covariate_spaces = {
    'centered': CovariateSpace(xmin = -1.0, xmax = 1.0)
}

# Adapted from scikit-learn GP example
# http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy.html
ground_truths = {
    'high_freq_sinusoid': GroundTruth(
        variance = 0.01,
        mean_fn = lambda x: np.sin(3 * 2 * np.pi * x[0]),
        noise_fn = lambda rng: rng.normal(0, np.sqrt(0.01)),
        name = 'sin_freq_3_noise_0.01'),
    'low_freq_sinusoid': GroundTruth(
        variance = 0.01,
        mean_fn = lambda x: np.sin(1 * 2 * np.pi * x[0]),
        noise_fn = lambda rng: rng.normal(0, np.sqrt(0.01)),
        name = 'sin_freq_1_noise_0.01')
}


class UniformSelector():
    def __init__(self, covariate_space, rng):
        self.covariate_space = covariate_space
        self.rng = rng
        self.name = "uniform"

    # returns a numpy array
    def next_x(self, gp):
        return self.covariate_space.sample(1, self.rng)


class VarianceMinimizingSelector():
    def __init__(self, covariate_space, rng, num_xi=500, num_x_star=300):
        self.covariate_space = covariate_space
        self.rng = rng
        self.num_xi = num_xi
        self.num_x_star = num_x_star
        self.name = "varmin"

    # returns a numpy array
    def next_x(self, gp):
        # sample num_xi values of xi from covariate space (reused across all candidates x_*)
        # compute cholesky of K_n, then use cho_solve with every K(X_n, xi) vector
        #   O(n^3), then O(n^2)
        #   collect K(X_n, xi) vectors into a matrix?
        # 
        # sample num_x_star values of x_* from covariate space
        # for every x_* candidate:
        #    compute product of cho_solve result above with K(x_*, X_n)
        #    compute predictive variance (easier to do this for all x_* in batch?)
        #      should I use return_std code in sklearn or reimplement it?
        #    compute K(xi, x_*) (ditto above question, vectorized)
        #    average over all xi to get objective function value for this x_*
        
        # compare to runtime for computing inv(K_n) and doing matrix multiplications
        #   O(n^3), then O(n^2)
        
        # compute inverse kernel matrix
        K_n = gp.kernel_(gp.X_train_)
        L_n = cholesky(K_n, lower=True)
        L_n_inv = solve_triangular(L_n, np.eye(L_n.shape[0]), lower=True)
        K_n_inv = L_n_inv.T.dot(L_n_inv)

        # redundant check, time alternatives
        L_n_inv2 = solve_triangular(L_n.T, np.eye(L_n.shape[0]))
        K_n_inv2 = L_n_inv2.dot(L_n_inv2.T)
        assert np.allclose(L_n_inv, L_n_inv2.T)
        assert np.allclose(K_n_inv, K_n_inv2)

        # sample xi, compute kernel products
        all_xi = self.covariate_space.sample(self.num_xi, self.rng)[:, np.newaxis]
        K_n_trans_xi = gp.kernel_(gp.X_train_, all_xi)  # n x num_xi
        K_n_inv_xi_prod = np.dot(K_n_inv, K_n_trans_xi)  # n x num_xi

        # redundant check, time alternatives
        K_n_inv_xi_prod2 = np.einsum("ij,jk->ik", K_n_inv, K_n_trans_xi)
        assert np.allclose(K_n_inv_xi_prod, K_n_inv_xi_prod2)

        # sample x_star, compute kernel products
        all_x_star = self.covariate_space.sample(self.num_x_star, self.rng)[:, np.newaxis]
        K_n_trans_x_star = gp.kernel_(gp.X_train_, all_x_star)  # n x num_x_star
        qform_xi_x_star = K_n_inv_xi_prod.T.dot(K_n_trans_x_star)  # num_xi x num_x_star

        # redundant check, time alternatives
        qform_xi_x_star2 = np.einsum("ij,ik -> jk", K_n_inv_xi_prod, K_n_trans_x_star)
        assert np.allclose(qform_xi_x_star, qform_xi_x_star2)

        # compute predictive variance
        var_x_star = gp.kernel_.diag(all_x_star)  # num_x_star
        var_x_star -= np.einsum("ik,jk,ij->k", K_n_trans_x_star, K_n_trans_x_star, K_n_inv)
        assert not np.any(var_x_star < 1e-8)

        # redundant check, time alternatives
        _, std_x_star2 = gp.predict(all_x_star, return_std=True)
        assert np.allclose(var_x_star, std_x_star2 ** 2)

        # compute average variance reduction at each x_star
        var_delta = np.power(qform_xi_x_star - gp.kernel_(all_xi, all_x_star), 2) # num_xi x num_x_star
        avg_var_delta = np.mean(var_delta, axis=0)  # num_x_star (average across xi)
        avg_var_delta /= var_x_star
        x_star_index = np.argmax(avg_var_delta)

        # print np.vstack([all_x_star[:,0], avg_var_delta]).T
        print all_x_star[x_star_index]
        return all_x_star[x_star_index]


# Compute E(y_hat - y)^2 = E(y_hat - f)^2 + E(f - y)^2
#                        = E(y_hat - f)^2 + sigma^2
# where y_hat is the posterior mean, f is the ground truth mean
# and y is a new realization incorporating independent noise.
#
# Note MSE measures how well the posterior mean predicts the ground truth mean,
# but not how well the posterior confidence envelope matches a confidence envelope
# around the ground truth
def compute_mse(gp, covariate_space, ground_truth, rng):
    N = 1000
    X_pred = covariate_space.sample(N, rng)[:, np.newaxis]
    y_hat = gp.predict(X_pred)
    y = np.array([ ground_truth.observe_y_fn(x, rng) for x in X_pred ])
    return np.mean((y_hat - y) ** 2) + ground_truth.variance


def learn_gp(x_selector, kernel, update_theta,
             covariate_space, ground_truth, 
             obs_rng, eval_rng, 
             N_init, N_final, N_eval_pts):
    
    print "\nLearning GP"
    print "Initial kernel: ", kernel

    if update_theta:
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0)
    else:
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, optimizer=None)
        # setting optimizer to None will prevent hyperparameter fitting

    # generate first N observations randomly
    X = covariate_space.sample(N_init-1, x_selector.rng)[:, np.newaxis]
    y = np.array([ground_truth.observe_y_fn(x, obs_rng) for x in X])

    gp.fit(X, y)

    theta_iterates = np.empty((0,kernel.theta.size))

    eval_indices = np.linspace(N_init, N_final, num=N_eval_pts, dtype=int)
    mse_values = []

    posterior_plot = PosteriorPlot(covariate_space, ground_truth.mean_fn, N_eval_pts)

    for i in range(N_init, N_final+1):
        x_star = x_selector.next_x(gp)
        y_star = ground_truth.observe_y_fn(x_star, obs_rng)

        # update data
        X = np.vstack([X, x_star])
        y = np.append(y, y_star)

        assert X.shape[0] == i, y.size == i

        # update hyperparameters
        gp.fit(X, y)

        theta_iterates = np.vstack([theta_iterates, np.exp(gp.kernel_.theta)])
        # should only track theta_iterates at eval_indices?

        if i in eval_indices:
            mse = compute_mse(gp, covariate_space, ground_truth, eval_rng)
            mse_values.append(mse)
            print "%d points, MSE: %f" % (i, mse)

            plot_num = np.where(eval_indices == i)[0][0]
            posterior_plot.append(gp, X, y, plot_num, n_points=i)

    print "Final kernel: ", gp.kernel_

    gen_filename = lambda fig_type: filename_for(x_selector.name, ground_truth.name,
        N_init, N_final, fig_type)

    if update_theta:
        posterior_plot.save(gen_filename("posterior_seq"))
        plot_mse(eval_indices, mse_values, ground_truth.variance, gen_filename("mse_seq"))
        plot_log_marginal_likelihood(gp, theta_iterates, gen_filename("lml_seq"))
    else:
        kernel_str = '_'.join(map(lambda f: "%.2e" % f, np.exp(kernel.theta)))
        posterior_plot.save(gen_filename("posterior_" + kernel_str))

    return (eval_indices, mse_values)


if __name__ == "__main__":
    print "Called with arguments:"
    print sys.argv

    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', choices=['uniform', 'varmin'], required=True)
    parser.add_argument('--fix-theta', action='store_true')
    parser.add_argument('--covariate-space', choices=covariate_spaces.keys(), required=True)
    parser.add_argument('--ground-truth', choices=ground_truths.keys(), required=True)
    parser.add_argument('--nmin', type=int, default=11)
    parser.add_argument('--nmax', type=int, default=20)
    parser.add_argument('--n_eval_pts', type=int, default=10)
    parser.add_argument('--random-seed', type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(fig_prefix):
        os.makedirs(fig_prefix)

    covariate_space = covariate_spaces[args.covariate_space]
    ground_truth = ground_truths[args.ground_truth]

    selector_rng = np.random.RandomState(args.random_seed)
    if args.strategy == 'uniform':
        x_selector = UniformSelector(covariate_space, selector_rng)
    elif args.strategy == 'varmin':
        x_selector = VarianceMinimizingSelector(covariate_space, selector_rng)

    default_kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-10, 1e+1))

    if args.fix_theta:
        # learn GP with different fixed values of theta

        learn_gp_fix_kernel = lambda kernel: learn_gp(
            x_selector = x_selector,
            kernel = kernel,
            update_theta = False,
            covariate_space = covariate_space,
            ground_truth = ground_truth,
            obs_rng = np.random.RandomState(args.random_seed),
            eval_rng = np.random.RandomState(args.random_seed),
            N_init = args.nmin,
            N_final = args.nmax,
            N_eval_pts = args.n_eval_pts
        )

        # try different length-scale values
        mse_diffls_plot = MSEPlot(ground_truth.variance,
            title="Learning GP with fixed hyperparameters, variance=%.3f" % ground_truth.variance)

        for length_scale in ground_truth.approx_length_scale * np.logspace(-1, 1, 9):
            kernel = RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e3)) \
                + WhiteKernel(noise_level=ground_truth.variance)
            eval_indices, mse_values = learn_gp_fix_kernel(kernel)
            mse_diffls_plot.append(eval_indices, mse_values, 
                label="length-scale=%.2e" % length_scale)

        mse_filename = filename_for(x_selector.name, ground_truth.name,
            args.nmin, args.nmax, "mse_diffls")
        mse_diffls_plot.save(mse_filename)

        # try different variance values
        mse_diffvar_plot = MSEPlot(ground_truth.variance, 
            title="Learning GP with fixed hyperparameters, length-scale=%.3f" % ground_truth.approx_length_scale)

        for variance in ground_truth.variance * np.logspace(-2, 2, 5):
            kernel = RBF(length_scale=ground_truth.approx_length_scale) \
                + WhiteKernel(noise_level=variance, noise_level_bounds=(1e-10, 1e+1))
            eval_indices, mse_values = learn_gp_fix_kernel(kernel)
            mse_diffvar_plot.append(eval_indices, mse_values, 
                label="variance=%.2e" % variance)

        mse_filename = filename_for(x_selector.name, ground_truth.name,
            args.nmin, args.nmax, "mse_diffvar")
        mse_diffvar_plot.save(mse_filename)

    else:
        # run sequential version
        learn_gp(
            x_selector = x_selector,
            kernel = default_kernel,
            update_theta = True,
            covariate_space = covariate_space,
            ground_truth = ground_truth,
            obs_rng = np.random.RandomState(args.random_seed),
            eval_rng = np.random.RandomState(args.random_seed),
            N_init = args.nmin,
            N_final = args.nmax,
            N_eval_pts = args.n_eval_pts
        )
