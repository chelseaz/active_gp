#!/usr/bin/python

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from operator import itemgetter


class IncrementalPlot():
    def complete(self):
        pass

    def save(self, filename):
        self.complete()
        self.fig.savefig(filename)
        plt.close(self.fig) 


class MSEPlot(IncrementalPlot):
    def __init__(self, true_variance):
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.ax.axhline(true_variance, color='r', linewidth=1)
        
    def append(self, n_values, mse_values, label):
        self.ax.plot(n_values, mse_values, label=label, linewidth=1)
        # TODO: different colors

    def complete(self):
        self.ax.set_yscale("log")
        self.ax.set_xlabel("Number of training points")
        self.ax.set_ylabel("MSE")
        self.ax.set_title("MSE of posterior mean across iterations")
        self.ax.legend()


class PosteriorPlot(IncrementalPlot):
    def __init__(self, covariate_space, truth_fn, n_plots):
        self.X_ = np.linspace(covariate_space.xmin, covariate_space.xmax, 100)
        self.X__matrix = self.X_[:, np.newaxis]
        self.y_truth = [ truth_fn(x) for x in self.X__matrix ]

        self.ncols = int(np.floor(np.sqrt(n_plots)))
        self.nrows = int(np.ceil(n_plots / float(self.ncols)))
        self.fig, self.axarr = plt.subplots(self.nrows, self.ncols, figsize=(20, 20))
        self.fig.subplots_adjust(hspace = 0.25)


    def append(self, gp, X_train, y_train, plot_num, n_points):
        plot_row = plot_num / self.ncols
        plot_col = plot_num - plot_row * self.ncols
        ax = self.axarr[plot_row, plot_col]

        # Plot GP posterior
        y_mean, y_cov = gp.predict(self.X__matrix, return_cov=True)
        ax.plot(self.X_, y_mean, 'k', lw=2, zorder=9)
        ax.fill_between(self.X_, y_mean - np.sqrt(np.diag(y_cov)),
                        y_mean + np.sqrt(np.diag(y_cov)),
                        alpha=0.5, color='k')

        # Plot ground truth
        ax.plot(self.X_, self.y_truth, 'r', lw=2, zorder=9)

        # Draw training points, highlighting the last point
        ax.scatter(X_train[:-1, 0], y_train[:-1], c='k', s=20, zorder=10)
        ax.scatter(X_train[-1:, 0], y_train[-1:], c='r', s=30, zorder=11, edgecolors=(0, 0, 0))
        
        ax.set_title("With %d training points:\nLog-Marginal-Likelihood: %s"
                     % (n_points, gp.log_marginal_likelihood(gp.kernel_.theta)))



# Adapted from http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy.html
def plot_posterior(gp, X_train, y_train, covariate_space, truth_fn, filename):
    plt.figure(figsize=(10, 6))
    X_ = np.linspace(covariate_space.xmin, covariate_space.xmax, 100)
    X__matrix = X_[:, np.newaxis]

    # Plot GP posterior
    y_mean, y_cov = gp.predict(X__matrix, return_cov=True)
    plt.plot(X_, y_mean, 'k', lw=2, zorder=9)
    plt.fill_between(X_, y_mean - np.sqrt(np.diag(y_cov)),
                     y_mean + np.sqrt(np.diag(y_cov)),
                     alpha=0.5, color='k')

    # Plot ground truth
    y_truth = [ truth_fn(x) for x in X__matrix ]
    plt.plot(X_, y_truth, 'r', lw=2, zorder=9)

    # Draw training points, highlighting the last point
    plt.scatter(X_train[:-1, 0], y_train[:-1], c='k', s=20, zorder=10)
    plt.scatter(X_train[-1:, 0], y_train[-1:], c='r', s=30, zorder=11, edgecolors=(0, 0, 0))
    
    plt.title("Optimum: %s\nLog-Marginal-Likelihood: %s"
              % (gp.kernel_, gp.log_marginal_likelihood(gp.kernel_.theta)))
    plt.savefig(filename)
    plt.close()


# Adapted from http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy.html
def plot_log_marginal_likelihood(gp, theta_iterates, filename):
    plt.figure()
    theta0 = np.logspace(-2, 3, 49)
    theta1 = np.logspace(-3, 0, 50)
    Theta0, Theta1 = np.meshgrid(theta0, theta1)
    LML = [[gp.log_marginal_likelihood(np.log([Theta0[i, j], Theta1[i, j]]))
            for i in range(Theta0.shape[0])] for j in range(Theta0.shape[1])]
    LML = np.array(LML).T

    # contour plot requires positive values
    # If necessary, subtract a constant from log likelihoods so all are negative
    # Then negate log likelihoods
    LML = LML - max(LML.max()+10, 0)
    vmin, vmax = (-LML).min(), (-LML).max()
    # print "vmin:", vmin, "vmax:", vmax
    # set vmax to weighted geometric mean
    vmax = np.exp(0.5*np.log(vmin) + 0.5*np.log(vmax))

    level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 50), decimals=1)
    plt.contour(Theta0, Theta1, -LML,
                levels=level, norm=LogNorm(vmin=vmin, vmax=vmax))
    plt.colorbar()

    # plot iterates
    x = list(map(itemgetter(0), theta_iterates))
    y = list(map(itemgetter(1), theta_iterates))
    # plt.scatter(x, y, c='k', s=20)
    plt.plot(x, y, 'k-', linewidth=1)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Length-scale")
    plt.ylabel("Noise level")
    plt.title("Negative log marginal likelihood")
    plt.savefig(filename)
    plt.close()


def plot_mse(all_n, all_mse, true_variance, filename):
    plt.figure()

    plt.plot(all_n, all_mse, 'k-', linewidth=1)
    plt.axhline(true_variance, color='r', linewidth=1)
    
    plt.yscale("log")
    
    plt.title("MSE of posterior mean across iterations")
    plt.savefig(filename)
    plt.close()
