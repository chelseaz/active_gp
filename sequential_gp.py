#!/usr/bin/python

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from viz import *

rng = np.random.RandomState(0)


def sinusoid_with_noise(x):
    # Used in scikit-learn GP example
    # http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy.html
    return 0.5 * np.sin(3 * x[0]) + rng.normal(0, np.sqrt(0.25))

def sinusoid_noiseless(x):
    return 0.5 * np.sin(3 * x[0])

def uniform_sampling():
    return rng.uniform(0, 5, 1)


# select_x_fn returns a numpy array
# observe_y_fn takes in a numpy array and returns a scalar output
def run_sequential(select_x_fn, observe_y_fn, truth_fn):
    N_init = 11
    N_final = 20

    # generate first N observations randomly
    X = rng.uniform(0, 5, N_init-1)[:, np.newaxis]
    y = np.array([observe_y_fn(x) for x in X])

    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-10, 1e+1))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0).fit(X, y)

    theta_iterates = np.empty((0,kernel.theta.size))

    for i in range(N_init, N_final+1):
        x_star = select_x_fn()
        y_star = observe_y_fn(x_star)

        # update data
        X = np.vstack([X, x_star])
        y = np.append(y, y_star)

        assert X.shape[0] == i, y.size == i

        # update hyperparameters
        gp.fit(X, y)

        theta_iterates = np.vstack([theta_iterates, np.exp(gp.kernel_.theta)])

        print "With %d points" % i
        print "final kernel: ", gp.kernel_
        plot_posterior(gp, X, y, truth_fn, "posterior_%d.png" % i)
        
    plot_log_marginal_likelihood(gp, theta_iterates, "lml_%d_%d.png" % (N_init, N_final))


if __name__ == "__main__":
    run_sequential(
        select_x_fn=uniform_sampling,
        observe_y_fn=sinusoid_with_noise,
        truth_fn=sinusoid_noiseless
    )
