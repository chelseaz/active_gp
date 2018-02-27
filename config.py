#!/usr/bin/python

import numpy as np

length_scale_bounds=(1e-2, 1e3)
noise_level_bounds=(1e-5, 1e+1)


class UniformCovariateSpace():
    def __init__(self, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax
        self.name = 'uniform'

    # returns a nx1 numpy array
    def sample(self, n, rng):
        return rng.uniform(self.xmin, self.xmax, n)[:, np.newaxis]

    def dimension(self):
        return 1


class GaussianCovariateSpace():
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.xmin = -3*scale
        self.xmax = 3*scale
        self.name = 'gaussian'

    # returns a nx1 numpy array
    def sample(self, n, rng):
        return rng.normal(self.loc, self.scale, n)[:, np.newaxis]

    def dimension(self):
        return 1


class MVGaussianCovariateSpace():
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        self.xmin = -3*np.diag(cov)
        self.xmax = 3*np.diag(cov)
        self.name = 'mvgaussian'

    # returns a nxd numpy array
    def sample(self, n, rng):
        return rng.multivariate_normal(self.mean, self.cov, n)

    def dimension(self):
        return self.mean.size


class GroundTruth():
    # mean_fn takes in a numpy array and returns a scalar output
    # noise_fn takes in a random state and returns a scalar output
    def __init__(self, variance, mean_fn, noise_fn, name, approx_length_scale):
        self.variance = variance
        self.mean_fn = mean_fn
        self.noise_fn = noise_fn
        self.name = name
        self.observe_y_fn = lambda x, rng: mean_fn(x) + noise_fn(rng)

        # TODO: calculate this programmatically
        self.approx_length_scale = approx_length_scale


covariate_spaces = {
    'uniform': UniformCovariateSpace(xmin = -1.0, xmax = 1.0),
    'gaussian': GaussianCovariateSpace(loc = 0, scale = 1.0/3),  
    # note need scale = 1/sqrt(3) for same variance as above
    'gaussian2d': MVGaussianCovariateSpace(mean = np.zeros(2), cov = np.eye(2)),
    'gaussian5d': MVGaussianCovariateSpace(mean = np.zeros(5), cov = np.eye(5)),
    'gaussian10d': MVGaussianCovariateSpace(mean = np.zeros(10), cov = np.eye(10))
}

ground_truths = {
    'high_freq_sinusoid': GroundTruth(
        variance = 0.25,
        mean_fn = lambda x: np.sin(3 * 2 * np.pi * x[0]),
        noise_fn = lambda rng: rng.normal(0, np.sqrt(0.25)),
        name = 'sin_freq_3_noise_0.25',
        approx_length_scale = None),
    'low_freq_sinusoid': GroundTruth(
        variance = 0.25,
        mean_fn = lambda x: np.sin(1 * 2 * np.pi * x[0]),
        noise_fn = lambda rng: rng.normal(0, np.sqrt(0.25)),
        name = 'sin_freq_1_noise_0.25',
        approx_length_scale = 0.334),
    'paraboloid': GroundTruth(
        variance = 0.25,
        mean_fn = lambda x: x[0]**2 + x[1]**2,
        noise_fn = lambda rng: rng.normal(0, np.sqrt(0.25)),
        name = 'paraboloid_1_noise_0.25',
        approx_length_scale = 1.0),
    '5d_paraboloid': GroundTruth(
        variance = 0.25,
        mean_fn = lambda x: 0.2*np.power(x, 2).sum(),
        noise_fn = lambda rng: rng.normal(0, np.sqrt(0.25)),
        name = '5d_paraboloid_0.2_noise_0.25',
        approx_length_scale = 3.17),
    '10d_paraboloid': GroundTruth(
        variance = 0.25,
        mean_fn = lambda x: 0.1*np.power(x, 2).sum(),
        noise_fn = lambda rng: rng.normal(0, np.sqrt(0.25)),
        name = '10d_paraboloid_0.1_noise_0.25',
        approx_length_scale = 4.15)
}
