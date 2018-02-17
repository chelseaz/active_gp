#!/usr/bin/python

import argparse
import numpy as np
import os
import sys

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from evaluate import Evaluator
from selection import RandomSelector, VarianceMinimizingSelector
from viz import *

fig_prefix = "figs/"

def filename_for(strategy, update_theta, ground_truth, covariate_space, N_init, N_final, custom, extension="png"):
    if update_theta is None:
        fixed_or_est = "both"
    elif update_theta:
        fixed_or_est = "estimated"
    else:
        fixed_or_est = "fixed"

    return fig_prefix + "%s_%s_%s_%s_%s_%d_%d.%s" % \
        (ground_truth, covariate_space, strategy, fixed_or_est, custom, N_init, N_final, extension)

def kernel_to_str(kernel):
    return '_'.join(map(lambda f: "%.2e" % f, np.exp(kernel.theta)))


class UniformCovariateSpace():
    def __init__(self, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax
        self.name = 'uniform'

    def sample(self, n, rng):
        return rng.uniform(self.xmin, self.xmax, n)


class GaussianCovariateSpace():
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.xmin = -3*scale
        self.xmax = 3*scale
        self.name = 'gaussian'

    def sample(self, n, rng):
        return rng.normal(self.loc, self.scale, n)


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
    'uniform': UniformCovariateSpace(xmin = -1.0, xmax = 1.0),
    'gaussian': GaussianCovariateSpace(loc = 0, scale = 1.0/3)
}

# Adapted from scikit-learn GP example
# http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy.html
ground_truths = {
    'high_freq_sinusoid': GroundTruth(
        variance = 0.25,
        mean_fn = lambda x: np.sin(3 * 2 * np.pi * x[0]),
        noise_fn = lambda rng: rng.normal(0, np.sqrt(0.25)),
        name = 'sin_freq_3_noise_0.25'),
    'low_freq_sinusoid': GroundTruth(
        variance = 0.25,
        mean_fn = lambda x: np.sin(1 * 2 * np.pi * x[0]),
        noise_fn = lambda rng: rng.normal(0, np.sqrt(0.25)),
        name = 'sin_freq_1_noise_0.25')
}



def learn_gp(x_selector, kernel, update_theta,
             covariate_space, ground_truth, 
             selector_rng, obs_rng, eval_rng, 
             N_init, N_final, N_eval_pts, plot_all):
    
    print "\nLearning GP"
    print "Initial kernel: ", kernel

    if update_theta:
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0)
    else:
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, optimizer=None)
        # setting optimizer to None will prevent hyperparameter fitting

    # generate first N observations randomly
    X = covariate_space.sample(N_init-1, selector_rng)[:, np.newaxis]
    y = np.array([ground_truth.observe_y_fn(x, obs_rng) for x in X])

    gp.fit(X, y)

    theta_iterates = np.empty((0,kernel.theta.size))

    evaluator = Evaluator(covariate_space, ground_truth, N_init, N_final, N_eval_pts)

    posterior_plot = PosteriorPlot(covariate_space, ground_truth.mean_fn, N_eval_pts)
    posterior_animation = PosteriorAnimation()
    density_plot = DensityPlot(N_eval_pts)
    if isinstance(x_selector, VarianceMinimizingSelector):
        objective_plot = ObjectivePlot(N_eval_pts)
    else:
        objective_plot = None

    for i in range(N_init, N_final+1):
        x_star = x_selector.next_x(gp, selector_rng)
        y_star = ground_truth.observe_y_fn(x_star, obs_rng)

        # update data
        X = np.vstack([X, x_star])
        y = np.append(y, y_star)

        assert X.shape[0] == i, y.size == i

        # update hyperparameters
        gp.fit(X, y)

        theta_iterates = np.vstack([theta_iterates, np.exp(gp.kernel_.theta)])
        # should only track theta_iterates at eval_indices?

        which_eval_index = evaluator.evaluate(i, gp, eval_rng)
        if which_eval_index is not None:
            plot_num = which_eval_index
            y_mean, y_cov = posterior_plot.append(gp, X, y, plot_num)
            posterior_animation.append(i, y_mean, y_cov)

            density_plot.append(X, plot_num)
            if objective_plot is not None:
                objective_plot.append(x_selector, plot_num, n_points=i)

    print "Final kernel: ", gp.kernel_

    def gen_filename(fig_type, extension="png"): 
        return filename_for(x_selector.name, update_theta, ground_truth.name, covariate_space.name, 
            N_init, N_final, fig_type, extension=extension)

    kernel_str = kernel_to_str(kernel)

    posterior_plot.save(gen_filename("posterior_" + kernel_str))
    density_plot.save(gen_filename("training_density_" + kernel_str))
    if objective_plot is not None:
        objective_plot.save(gen_filename("objective_" + kernel_str))
    if plot_all:
        plot_log_marginal_likelihood(gp, theta_iterates, gen_filename("lml_" + kernel_str))

        eval_plot = EvalPlot(ground_truth.variance,
            title="Learning GP with initial hyperparameters %s" % kernel)
        eval_plot.append(evaluator, label="estimated")
        eval_plot.save(gen_filename("eval_" + kernel_str))

    posterior_animation.set_quantities(X, y, posterior_plot.X_, posterior_plot.y_truth)
    posterior_animation.save(gen_filename("posterior_" + kernel_str, extension="gif"))

    return evaluator


if __name__ == "__main__":
    print "Called with arguments:"
    print sys.argv

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['seq', 'compare-seq-fixed', 'compare-seq-seq', 'compare-fixed-fixed'], default='seq')
    parser.add_argument('--strategy', choices=['random', 'varmin'], required=True)
    parser.add_argument('--covariate-space', choices=covariate_spaces.keys(), required=True)
    parser.add_argument('--ground-truth', choices=ground_truths.keys(), required=True)
    parser.add_argument('--nmin', type=int, default=11)
    parser.add_argument('--nmax', type=int, default=20)
    parser.add_argument('--n-eval-pts', type=int, default=12)
    parser.add_argument('--random-seed', type=int, default=123)
    args = parser.parse_args()

    if not os.path.exists(fig_prefix):
        os.makedirs(fig_prefix)

    covariate_space = covariate_spaces[args.covariate_space]
    ground_truth = ground_truths[args.ground_truth]

    # make these wrappers to standardize the random number generator across runs
    create_selector_rng = lambda: np.random.RandomState(args.random_seed)
    create_obs_rng = lambda: np.random.RandomState(args.random_seed*2)
    create_eval_rng = lambda: np.random.RandomState(args.random_seed*4)

    if args.strategy == 'random':
        x_selector = RandomSelector(covariate_space)
    elif args.strategy == 'varmin':
        x_selector = VarianceMinimizingSelector(covariate_space)

    default_kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-10, 1e+1))

    def learn_gp_wrapper(kernel, update_theta, plot_all=False):
        return learn_gp(
            x_selector = x_selector,
            kernel = kernel,
            update_theta = update_theta,
            covariate_space = covariate_space,
            ground_truth = ground_truth,
            selector_rng = create_selector_rng(),
            obs_rng = create_obs_rng(),
            eval_rng = create_eval_rng(),
            N_init = args.nmin,
            N_final = args.nmax,
            N_eval_pts = args.n_eval_pts,
            plot_all = plot_all
        )

    def compare_theta_values(update_theta):
        fixed_or_est = "estimated" if update_theta else "fixed"

        # try different length-scale values
        eval_diffls_plot = EvalPlot(ground_truth.variance,
            title="Learning GP with %s hyperparameters, variance=%.3f" % (fixed_or_est, ground_truth.variance))

        for length_scale in ground_truth.approx_length_scale * np.logspace(-1, 1, 9):
            kernel = RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e3)) \
                + WhiteKernel(noise_level=ground_truth.variance)
            evaluator = learn_gp_wrapper(kernel, update_theta)
            eval_diffls_plot.append(evaluator, label="length-scale=%.2e" % length_scale)

        eval_filename = filename_for(x_selector.name, update_theta, 
            ground_truth.name, covariate_space.name,
            args.nmin, args.nmax, "eval_diffls")
        eval_diffls_plot.save(eval_filename)

        # try different variance values
        eval_diffvar_plot = EvalPlot(ground_truth.variance, 
            title="Learning GP with %s hyperparameters, length-scale=%.3f" % (fixed_or_est, ground_truth.approx_length_scale))

        for variance in ground_truth.variance * np.logspace(-2, 2, 5):
            kernel = RBF(length_scale=ground_truth.approx_length_scale) \
                + WhiteKernel(noise_level=variance, noise_level_bounds=(1e-10, 1e+1))
            evaluator = learn_gp_wrapper(kernel, update_theta)
            eval_diffvar_plot.append(evaluator, label="variance=%.2e" % variance)

        eval_filename = filename_for(x_selector.name, update_theta, 
            ground_truth.name, covariate_space.name,
            args.nmin, args.nmax, "eval_diffvar")
        eval_diffvar_plot.save(eval_filename)


    if args.mode == 'compare-fixed-fixed':
        # learn GP with different fixed values of theta
        compare_theta_values(update_theta=False)

    elif args.mode == 'compare-seq-seq':
        # learn GP with different initial values of theta, which are estimated over time
        compare_theta_values(update_theta=True)

    elif args.mode == 'compare-seq-fixed':
        # compare GP learning with fixed and estimated theta
        eval_plot = EvalPlot(ground_truth.variance,
            title="Learning GP with initial hyperparameters %s" % default_kernel)

        evaluator = learn_gp_wrapper(default_kernel, update_theta=False)
        eval_plot.append(evaluator, label="fixed")

        evaluator = learn_gp_wrapper(default_kernel, update_theta=True)
        eval_plot.append(evaluator, label="estimated")

        eval_filename = filename_for(x_selector.name, None,
            ground_truth.name, covariate_space.name,
            args.nmin, args.nmax, "eval")
        eval_plot.save(eval_filename)

    else:
        # run sequential version
        learn_gp_wrapper(default_kernel, update_theta=True, plot_all=True)
        