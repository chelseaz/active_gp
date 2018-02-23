#!/usr/bin/python

import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from operator import itemgetter

import config


class IncrementalAnimation(object):
    def __init__(self):
        self.iterates = []

    # assumes save has been called, so self.fig and self.ax are defined
    def init_anim(self):
        raise NotImplementedError

    # assumes save has been called, so self.fig and self.ax are defined
    def update_anim(self, iterate):
        raise NotImplementedError

    def save(self, filename, figsize=(8,6)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        anim = FuncAnimation(self.fig, self.update_anim, frames=self.iterates, init_func=self.init_anim,
            interval=1e3)
        anim.save(filename, dpi=80, writer='imagemagick')


class IncrementalAnimation3dPlotting(IncrementalAnimation):
    # override default save method to use 3d projection
    def save(self, filename, figsize=(8,6)):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        anim = FuncAnimation(self.fig, self.update_anim, frames=self.iterates, init_func=self.init_anim,
            interval=1e3)
        anim.save(filename, dpi=80, writer='imagemagick')


class PosteriorAnimation(IncrementalAnimation):
    def __init__(self, xlim, truth_fn, point_size=20):
        super(PosteriorAnimation, self).__init__()
        self.xlim = xlim
        self.set_quantities(truth_fn)
        self.point_size = point_size

    def set_quantities(self, truth_fn):
        xmin, xmax = self.xlim
        self.X_ = np.linspace(xmin, xmax, 100)
        self.X__matrix = self.X_[:, np.newaxis]
        self.y_truth = [ truth_fn(x) for x in self.X__matrix ]

    def append(self, n_points, gp, X_train, y_train):
        y_mean, y_cov = gp.predict(self.X__matrix, return_cov=True)
        self.iterates.append((n_points, y_mean, y_cov))

        # keep around the latest values
        self.X_train = X_train
        self.y_train = y_train

    # assumes save has been called, so self.fig and self.ax are defined
    def init_anim(self):
        self.ax.set_xlim(self.xlim)
        self.ax.plot(self.X_, self.y_truth, 'r', lw=2, zorder=9)
        self.posterior_mean, = self.ax.plot([], [], 'k', lw=2, zorder=9)
        self.posterior_interval = self.ax.fill_between([], [], [])
        self.points = self.ax.scatter(self.X_train, self.y_train, 
            s=np.zeros(self.y_train.size), c='k', edgecolors='k', zorder=10)
        # return self.posterior_mean, self.posterior_interval, self.points

    # assumes save has been called, so self.fig and self.ax are defined
    def update_anim(self, iterate):
        n_points, y_mean, y_cov = iterate
        total_n_points = self.y_train.size
        self.posterior_mean.set_data(self.X_, y_mean)

        # can't easily mutate existing posterior interval, so remove it and plot a new one
        self.posterior_interval.remove()
        self.posterior_interval = self.ax.fill_between(self.X_, y_mean - np.sqrt(np.diag(y_cov)),
            y_mean + np.sqrt(np.diag(y_cov)), alpha=0.5, color='k')

        point_sizes = [self.point_size] * (n_points-1) \
            + [self.point_size*2] \
            + [0] * (total_n_points-n_points)
        point_colors = ['k'] * total_n_points
        point_colors[n_points-1] = 'r'  # draw current point in red

        self.points.set_sizes(point_sizes)
        self.points.set_facecolors(point_colors)

        self.ax.set_title("GP posterior, %d points" % n_points)
        # return self.posterior_mean, self.posterior_interval, self.points


class Posterior2dAnimation(PosteriorAnimation, IncrementalAnimation3dPlotting):
    def __init__(self, xlim, ylim, truth_fn):
        self.ylim = ylim
        super(Posterior2dAnimation, self).__init__(xlim, truth_fn)

    def set_quantities(self, truth_fn):
        xmin, xmax = self.xlim
        ymin, ymax = self.ylim
        X, Y = np.meshgrid(np.linspace(xmin, xmax, 10), np.linspace(ymin, ymax, 10))
        self.X__matrix = np.vstack([X.ravel(), Y.ravel()]).T
        self.y_truth = [ truth_fn(x) for x in self.X__matrix ]

    # assumes save has been called, so self.fig and self.ax are defined
    def init_anim(self):
        pass
        # raise NotImplementedError

    # assumes save has been called, so self.fig and self.ax are defined
    def update_anim(self, iterate):
        pass
        # raise NotImplementedError


class DensityAnimation(IncrementalAnimation):
    def __init__(self, xlim):
        super(DensityAnimation, self).__init__()
        self.xlim = xlim

    def append(self, n_points, X_train):
        self.iterates.append(n_points)
        # keep around the latest values
        self.X_train = X_train

    # assumes save has been called, so self.fig and self.ax are defined
    def init_anim(self):
        pass

    # assumes save has been called, so self.fig and self.ax are defined
    def update_anim(self, iterate):
        n_points = iterate
        self.ax.clear()
        sns.distplot(self.X_train[:n_points, 0], rug=True, ax=self.ax)
        self.ax.set_xlim(self.xlim)
        self.ax.set_title("Histogram of first %d training points" % n_points)


class Density2dAnimation(DensityAnimation):
    def __init__(self, xlim, ylim):
        super(Density2dAnimation, self).__init__(xlim)
        self.ylim = ylim

    # assumes save has been called, so self.fig and self.ax are defined
    def update_anim(self, iterate):
        n_points = iterate
        x = self.X_train[:n_points, 0]
        y = self.X_train[:n_points, 1]

        self.ax.clear()
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.ax.scatter(x, y, c='k', marker='+')
        self.ax.set_title("Locations of first %d training points" % n_points)
        
        if n_points < 3: return
        sns.kdeplot(x, y, ax=self.ax)


class ObjectiveAnimation(IncrementalAnimation):
    def __init__(self, xlim):
        super(ObjectiveAnimation, self).__init__()
        self.xlim = xlim

    def append(self, n_points, selector):
        x = selector.all_x_star
        y = selector.avg_var_delta
        argmax_index = selector.x_star_index
        self.iterates.append((n_points, x, y, argmax_index))

    # assumes save has been called, so self.fig and self.ax are defined
    def init_anim(self):
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylabel("Variance reduction")
        self.objective, = self.ax.plot([], [], 'ko', markersize=5)
        self.objective_max, = self.ax.plot([], [], 'ro', markersize=10, markeredgecolor='k')

    # assumes save has been called, so self.fig and self.ax are defined
    def update_anim(self, iterate):
        n_points, x, y, argmax_index = iterate
        self.objective.set_data(x[:,0], y)
        self.objective_max.set_data([x[argmax_index,0]], [y[argmax_index]])
        # recompute and update y-axis limits
        self.ax.relim()
        self.ax.autoscale_view(scalex=False)
        self.ax.set_title("Selection of training point %d" % n_points)


class Objective2dAnimation(ObjectiveAnimation, IncrementalAnimation3dPlotting):
    def __init__(self, xlim, ylim):
        super(Objective2dAnimation, self).__init__(xlim)
        self.ylim = ylim

    # assumes save has been called, so self.fig and self.ax are defined
    def init_anim(self):
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)

    # assumes save has been called, so self.fig and self.ax are defined
    def update_anim(self, iterate):
        n_points, x, y, argmax_index = iterate
        self.ax.collections = []
        # the following does not respect zorder. scatter always ends up below trisurf.
        self.ax.plot_trisurf(x[:,0], x[:,1], y, cmap=plt.cm.viridis)
        self.ax.scatter([x[argmax_index,0]], [x[argmax_index,1]], [y[argmax_index]], c='r')
        self.ax.set_title("Selection of training point %d" % n_points, y=1.05)


class LMLAnimation(IncrementalAnimation):
    def __init__(self):
        super(LMLAnimation, self).__init__()

        theta0_min, theta0_max = np.log10(config.length_scale_bounds)
        theta1_min, theta1_max = np.log10(config.noise_level_bounds)
        theta0 = np.logspace(theta0_min, theta0_max, 49)
        theta1 = np.logspace(theta1_min, theta1_max, 50)
        self.Theta0, self.Theta1 = np.meshgrid(theta0, theta1)

    def append(self, n_points, gp, theta):
        LML = [[gp.log_marginal_likelihood(np.log([self.Theta0[i, j], self.Theta1[i, j]]))
            for i in range(self.Theta0.shape[0])] for j in range(self.Theta0.shape[1])]
        LML = np.array(LML).T
        self.iterates.append((n_points, LML, theta))

    # assumes save has been called, so self.fig and self.ax are defined
    def init_anim(self):
        self.ax.set_xscale("log")
        self.ax.set_yscale("log")
        self.ax.set_xlabel("Length-scale")
        self.ax.set_ylabel("Noise level")
        self.ax.set_title("Negative log marginal likelihood")

        self.fig.subplots_adjust(right = 0.8)
        self.cbar_ax = self.fig.add_axes([0.83, 0.1, 0.03, 0.8])

        self.theta_iterates, = self.ax.plot(np.empty(0), np.empty(0), 'k-', linewidth=1)
        self.current_theta, = self.ax.plot([], [], 'ro', markeredgecolor='k')
        # return self.theta_iterates,

    # assumes save has been called, so self.fig and self.ax are defined
    def update_anim(self, iterate):
        n_points, LML, (theta0, theta1) = iterate

        # contour plot requires positive values
        # If necessary, subtract a constant from log likelihoods so all are negative
        # Then negate log likelihoods
        LML = LML - max(LML.max()+10, 0)
        vmin, vmax = (-LML).min(), (-LML).max()
        # print "vmin:", vmin, "vmax:", vmax
        # # set vmax to weighted geometric mean
        # vmax = np.exp(0.5*np.log(vmin) + 0.5*np.log(vmax))

        level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 50), decimals=1)
        norm = LogNorm(vmin=vmin, vmax=vmax)

        # can't mutate existing contours, so remove them and plot new ones
        self.ax.collections = []
        contours = self.ax.contour(self.Theta0, self.Theta1, -LML, levels=level, norm=norm)

        # we need to remove and create the colorbar every time inside set axes
        # otherwise creating a new colorbar would eat away at the remaining space
        self.cbar_ax.clear()
        self.fig.colorbar(contours, cax=self.cbar_ax)

        self.theta_iterates.set_data(
            np.append(self.theta_iterates.get_xdata(), theta0),
            np.append(self.theta_iterates.get_ydata(), theta1)
        )
        self.current_theta.set_data([theta0], [theta1])

        self.ax.set_title("Negative log marginal likelihood, %d points" % n_points)
        # return self.theta_iterates,


class IncrementalPlot(object):
    def complete(self):
        pass

    def init_subplots(self, n_plots):
        self.ncols = int(np.floor(np.sqrt(n_plots)))
        self.nrows = int(np.ceil(n_plots / float(self.ncols)))
        self.fig, self.axarr = plt.subplots(self.nrows, self.ncols, figsize=(20, 20))
        self.fig.subplots_adjust(hspace = 0.25)

    def get_subplot(self, plot_num):
        plot_row = plot_num / self.ncols
        plot_col = plot_num - plot_row * self.ncols
        return self.axarr[plot_row, plot_col]

    def save(self, filename):
        self.complete()
        self.fig.savefig(filename)
        plt.close(self.fig) 


class EvalPlot(IncrementalPlot):
    def __init__(self, true_variance, title):
        self.title = title
        self.fig, (self.ax_mse, self.ax_var) = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.subplots_adjust(wspace = 0.25)
        # self.handles = []
        # self.labels = []

        self.ax_mse.axhline(true_variance, color='r', linewidth=1, linestyle='--')
        self.ax_var.axhline(true_variance, color='r', linewidth=1, linestyle='--')
        
    def append(self, evaluator, label):
        line, = self.ax_mse.plot(evaluator.eval_indices, evaluator.mse_values, label=label, linewidth=2)
        self.ax_var.plot(evaluator.eval_indices, evaluator.var_values, label=label, linewidth=2)
        # self.handles.append(line)
        # self.labels.append(label)

    def complete(self):
        self.ax_mse.set_yscale("log")
        self.ax_mse.set_xlabel("Number of training points")
        self.ax_mse.set_ylabel("MSE of posterior mean")

        self.ax_var.set_yscale("log")
        self.ax_var.set_xlabel("Number of training points")
        self.ax_var.set_ylabel("Expected posterior predictive variance")
        self.ax_var.legend()

        self.fig.suptitle(self.title)
        # self.fig.legend(self.handles, self.labels, loc='center right')


class DensityPlot(IncrementalPlot):
    def __init__(self, n_plots):
        self.init_subplots(n_plots)

    def append(self, plot_num, X_train):
        ax = self.get_subplot(plot_num)
        sns.distplot(X_train[:, 0], rug=True, ax=ax)
        ax.set_title("Histogram of first %d training points" % X_train.shape[0])


class ObjectivePlot(IncrementalPlot):
    def __init__(self, n_plots):
        self.init_subplots(n_plots)

    def append(self, plot_num, selector, n_points):
        x = selector.all_x_star[:,0]
        y = selector.avg_var_delta
        argmax_index = selector.x_star_index

        ax = self.get_subplot(plot_num)

        ax.scatter(x, y, c='k', s=10)
        # highlight the argmax
        ax.scatter(x[argmax_index], y[argmax_index], c='r', s=30, edgecolors=(0, 0, 0))

        ax.set_ylabel("Variance reduction")
        ax.set_title("Selection of training point %d" % n_points)


class PosteriorPlot(IncrementalPlot):
    def __init__(self, covariate_space, truth_fn, n_plots):
        self.X_ = np.linspace(covariate_space.xmin, covariate_space.xmax, 100)
        self.X__matrix = self.X_[:, np.newaxis]
        self.y_truth = [ truth_fn(x) for x in self.X__matrix ]

        self.init_subplots(n_plots)

    def append(self, plot_num, gp, X_train, y_train):
        y_mean, y_cov = gp.predict(self.X__matrix, return_cov=True)

        ax = self.get_subplot(plot_num)

        # Plot GP posterior
        ax.plot(self.X_, y_mean, 'k', lw=2, zorder=9)
        ax.fill_between(self.X_, y_mean - np.sqrt(np.diag(y_cov)),
                        y_mean + np.sqrt(np.diag(y_cov)),
                        alpha=0.5, color='k')

        # Plot ground truth
        ax.plot(self.X_, self.y_truth, 'r', lw=2, zorder=9)

        # Draw training points, highlighting the last point
        ax.scatter(X_train[:-1, 0], y_train[:-1], c='k', s=20, zorder=10)
        ax.scatter(X_train[-1:, 0], y_train[-1:], c='r', s=30, zorder=11, edgecolors=(0, 0, 0))
        
        ax.set_title("GP posterior with %d training points\nLog-Marginal-Likelihood: %s"
                     % (X_train.shape[0], gp.log_marginal_likelihood(gp.kernel_.theta)))


# Deprecated
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


# Deprecated
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


# Deprecated
def plot_mse(all_n, all_mse, true_variance, filename):
    plt.figure()

    plt.plot(all_n, all_mse, 'k-', linewidth=1)
    plt.axhline(true_variance, color='r', linewidth=1)
    
    plt.yscale("log")
    
    plt.title("MSE of posterior mean across iterations")
    plt.savefig(filename)
    plt.close()


# Deprecated
def plot_density(X, filename):
    plt.figure()
    
    # sns.kdeplot(X[:, 0])
    sns.distplot(X[:, 0], rug=True)

    plt.title("Distribution of training points")
    plt.savefig(filename)
    plt.close()
