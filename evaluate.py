#!/usr/bin/python

import numpy as np

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


class Evaluator():
    def __init__(self, covariate_space, ground_truth, N_init, N_final, N_eval_pts):
        self.covariate_space = covariate_space
        self.ground_truth = ground_truth
        self.eval_indices = np.linspace(N_init, N_final, num=N_eval_pts, dtype=int)
        self.mse_values = []

    def evaluate(self, point_num, gp, rng):
        if point_num in self.eval_indices:
            mse = compute_mse(gp, self.covariate_space, self.ground_truth, rng)
            self.mse_values.append(mse)
            print "%d points, MSE: %f" % (point_num, mse)

            which_eval_index = np.where(self.eval_indices == point_num)[0][0]
            return which_eval_index
        else:
            return None
