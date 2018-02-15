#!/usr/bin/python

import numpy as np

# Compute E(y_hat - y)^2 = E(y_hat - f)^2 + E(f - y)^2
#                        = E(y_hat - f)^2 + sigma^2
# where y_hat is the posterior mean, f is the ground truth mean
# and y is a new realization incorporating independent noise.
# Approximate by Monte Carlo.
#
# Note MSE measures how well the posterior mean predicts the ground truth mean,
# but not how well the posterior confidence envelope matches a confidence envelope
# around the ground truth
def compute_mse(y_hat, y, noise_variance):
    return np.mean((y_hat - y) ** 2) + noise_variance

# Compute predictive variance of the GP, integrated over covariate space
# Approximate by Monte Carlo.
def compute_var(y_std):
    return np.mean(y_std ** 2)


class Evaluator():
    def __init__(self, covariate_space, ground_truth, N_init, N_final, N_eval_pts):
        self.covariate_space = covariate_space
        self.ground_truth = ground_truth
        self.eval_indices = np.linspace(N_init, N_final, num=N_eval_pts, dtype=int)
        self.mse_values = []
        self.var_values = []

    def evaluate(self, point_num, gp, rng, n_sample=1000):
        if point_num in self.eval_indices:
            X_new = self.covariate_space.sample(n_sample, rng)[:, np.newaxis]

            y_hat, y_std = gp.predict(X_new, return_std=True)

            y = np.array([ self.ground_truth.observe_y_fn(x, rng) for x in X_new ])

            mse = compute_mse(y_hat, y, self.ground_truth.variance)
            var = compute_var(y_std)

            self.mse_values.append(mse)
            self.var_values.append(var)
            print """%d points\nMSE: %f\nPredictive variance: %f""" % (point_num, mse, var)

            which_eval_index = np.where(self.eval_indices == point_num)[0][0]
            return which_eval_index
        else:
            return None
