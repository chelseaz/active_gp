#!/usr/bin/python

import numpy as np

from scipy.linalg import cholesky, solve_triangular

class RandomSelector():
    def __init__(self, covariate_space):
        self.covariate_space = covariate_space
        self.name = "random"

    # returns a length d numpy array
    def next_x(self, gp, rng):
        return self.covariate_space.sample(1, rng)[0]


class VarianceMinimizingSelector():
    def __init__(self, covariate_space, num_xi=500, num_x_star=300):
        self.covariate_space = covariate_space
        self.num_xi = num_xi
        self.num_x_star = num_x_star
        self.name = "varmin"

    # returns a length d numpy array
    def next_x(self, gp, rng):
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

        # redundant check, takes about as long
        # L_n_inv2 = solve_triangular(L_n.T, np.eye(L_n.shape[0]))
        # K_n_inv2 = L_n_inv2.dot(L_n_inv2.T)
        # assert np.allclose(L_n_inv, L_n_inv2.T)
        # assert np.allclose(K_n_inv, K_n_inv2)

        # sample xi, compute kernel products
        all_xi = self.covariate_space.sample(self.num_xi, rng)
        K_n_trans_xi = gp.kernel_(gp.X_train_, all_xi)  # n x num_xi
        K_n_inv_xi_prod = np.dot(K_n_inv, K_n_trans_xi)  # n x num_xi

        # redundant check, slower
        # K_n_inv_xi_prod2 = np.einsum("ij,jk->ik", K_n_inv, K_n_trans_xi)
        # assert np.allclose(K_n_inv_xi_prod, K_n_inv_xi_prod2)

        # sample x_star, compute kernel products
        all_x_star = self.covariate_space.sample(self.num_x_star, rng)
        K_n_trans_x_star = gp.kernel_(gp.X_train_, all_x_star)  # n x num_x_star
        qform_xi_x_star = K_n_inv_xi_prod.T.dot(K_n_trans_x_star)  # num_xi x num_x_star

        # redundant check, slower
        # qform_xi_x_star2 = np.einsum("ij,ik -> jk", K_n_inv_xi_prod, K_n_trans_x_star)
        # assert np.allclose(qform_xi_x_star, qform_xi_x_star2)

        # compute predictive variance
        var_x_star = gp.kernel_.diag(all_x_star)  # num_x_star
        var_x_star -= np.einsum("ik,jk,ij->k", K_n_trans_x_star, K_n_trans_x_star, K_n_inv)
        assert not np.any(var_x_star < 1e-16)

        # redundant check, slower
        # _, std_x_star2 = gp.predict(all_x_star, return_std=True)
        # assert np.allclose(var_x_star, std_x_star2 ** 2)

        # compute average variance reduction at each x_star
        var_delta = np.power(qform_xi_x_star - gp.kernel_(all_xi, all_x_star), 2) # num_xi x num_x_star
        avg_var_delta = np.mean(var_delta, axis=0)  # num_x_star (average across xi)
        avg_var_delta /= var_x_star
        x_star_index = np.argmax(avg_var_delta)

        # save these quantities for access by plotting routines
        self.all_x_star = all_x_star
        self.avg_var_delta = avg_var_delta
        self.x_star_index = x_star_index

        return all_x_star[x_star_index]
