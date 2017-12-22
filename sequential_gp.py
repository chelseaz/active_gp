#!/usr/bin/python

import argparse
import numpy as np
import os
import sys

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from viz import *

fig_prefix = "figs/"


class CovariateSpace():
    def __init__(self, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax


class GroundTruth():
    # mean_fn takes in a numpy array and returns a scalar output
    # noise_fn takes in a random state and returns a scalar output
    def __init__(self, mean_fn, noise_fn, name):
        self.mean_fn = mean_fn
        self.noise_fn = noise_fn
        self.name = name
        self.observe_y_fn = lambda x, rng: mean_fn(x) + noise_fn(rng)


covariate_spaces = {
    'centered': CovariateSpace(xmin = -1.0, xmax = 1.0)
}

# Adapted from scikit-learn GP example
# http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy.html
ground_truths = {
    'high_freq_sinusoid': GroundTruth(
        mean_fn = lambda x: np.sin(3 * 2 * np.pi * x[0]),
        noise_fn = lambda rng: rng.normal(0, np.sqrt(0.01)),
        name = 'sin_freq_3_noise_0.01'),
    'low_freq_sinusoid': GroundTruth(
        mean_fn = lambda x: np.sin(1 * 2 * np.pi * x[0]),
        noise_fn = lambda rng: rng.normal(0, np.sqrt(0.01)),
        name = 'sin_freq_1_noise_0.01')
}

def uniform_sampling(covariate_space, rng):
    return rng.uniform(covariate_space.xmin, covariate_space.xmax, 1)


def compute_mse(gp, ground_truth):
    # sample x uniformly for now
    # eventually incorporate density of x
    pass


# select_x_fn returns a numpy array
def run_sequential(select_x_fn, covariate_space, ground_truth, 
                   rng, N_init, N_final):
    # generate first N observations randomly
    X = rng.uniform(covariate_space.xmin, covariate_space.xmax, N_init-1)[:, np.newaxis]
    y = np.array([ground_truth.observe_y_fn(x, rng) for x in X])

    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-10, 1e+1))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0).fit(X, y)

    theta_iterates = np.empty((0,kernel.theta.size))

    for i in range(N_init, N_final+1):
        x_star = select_x_fn(covariate_space, rng)
        y_star = ground_truth.observe_y_fn(x_star, rng)

        # update data
        X = np.vstack([X, x_star])
        y = np.append(y, y_star)

        assert X.shape[0] == i, y.size == i

        # update hyperparameters
        gp.fit(X, y)

        theta_iterates = np.vstack([theta_iterates, np.exp(gp.kernel_.theta)])

        print "With %d points" % i
        print "final kernel: ", gp.kernel_
        posterior_filename = fig_prefix + "%s_posterior_%d.png" % (ground_truth.name, i)
        plot_posterior(gp, X, y, covariate_space, ground_truth.mean_fn, posterior_filename)
        
    lml_filename = fig_prefix + "%s_lml_%d_%d.png" % (ground_truth.name, N_init, N_final)
    plot_log_marginal_likelihood(gp, theta_iterates, lml_filename)


if __name__ == "__main__":
    print "Called with arguments:"
    print sys.argv

    parser = argparse.ArgumentParser()
    parser.add_argument('--covariate_space', type=str, required=True)
    parser.add_argument('--ground_truth', type=str, required=True)
    parser.add_argument('--nmin', type=int, default=11)
    parser.add_argument('--nmax', type=int, default=20)
    parser.add_argument('--random_seed', type=int, default=0)
    args = parser.parse_args()

    try:
        covariate_space = covariate_spaces[args.covariate_space]
    except KeyError:
        print "Requested covariate space '%s' not found" % args.covariate_space
        sys.exit(2)

    try:
        ground_truth = ground_truths[args.ground_truth]
    except KeyError:
        print "Requested ground truth '%s' not found" % args.ground_truth
        sys.exit(2)

    if not os.path.exists(fig_prefix):
        os.makedirs(fig_prefix)

    run_sequential(
        select_x_fn = uniform_sampling,
        covariate_space = covariate_space,
        ground_truth = ground_truth,
        rng = np.random.RandomState(0),
        N_init = args.nmin,
        N_final = args.nmax
    )
