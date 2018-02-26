# GP active learning via predictive variance minimization 

In the following simulations, we assess how well GP active learning can learn a given response surface. We use a ground truth of a low-frequency sinusoid with independent Gaussian noise of variance 0.25. Training points are selected from a one-dimensional covariate space, normally distributed with mean 0 and variance 1/3.

To choose the next point, the active learning algorithm computes the expected reduction in predictive variance for a candidate training point by Monte Carlo, averaging over 500 query points. This objective is evaluated for 300 candidate training points. All query points and candidate training points are independently drawn from the covariate distribution. The training point with greatest expected variance reduction is chosen as the next training point, and we observe its output value.

The covariance function is usually unknown in practice. What happens if we use our active learning strategy with a fixed (and wrong) covariance function?

### Using active learning with a fixed covariance function (length-scale 1, noise variance 1)

#### Posterior
![](sin_freq_1_noise_0.25_gaussian_varmin_fixed_posterior_1.00e+00_1.00e+00_2_100.png)

#### Objective function computed by active learning algorithm
![](sin_freq_1_noise_0.25_gaussian_varmin_fixed_objective_1.00e+00_1.00e+00_2_100.png)

#### Distribution of selected training points
![](sin_freq_1_noise_0.25_gaussian_varmin_fixed_training_density_1.00e+00_1.00e+00_2_100.png)

### Estimating the covariance function results in a better posterior

Now we re-estimate the covariance hyperparameters after every new (x,y) observation by maximizing the marginal likelihood of the GP. 

#### Posterior
![](sin_freq_1_noise_0.25_gaussian_varmin_estimated_posterior_1.00e+00_1.00e+00_2_100.png)

#### Objective function computed by active learning algorithm
![](sin_freq_1_noise_0.25_gaussian_varmin_estimated_objective_1.00e+00_1.00e+00_2_100.png)

#### Distribution of selected training points
![](sin_freq_1_noise_0.25_gaussian_varmin_estimated_training_density_1.00e+00_1.00e+00_2_100.png)

#### Path of hyperparameter estimates in log-marginal-likelihood landscape at final iteration
![](sin_freq_1_noise_0.25_gaussian_varmin_estimated_lml_1.00e+00_1.00e+00_2_100.png)

#### The posterior after estimating hyperparameters has lower MSE
![](sin_freq_1_noise_0.25_gaussian_varmin_both_mse_2_100.png)

---

### How bad is the active learning strategy under different fixed hyperparameter settings?

#### With hyperparameters fixed at different length-scale settings, correct noise variance
![](sin_freq_1_noise_0.25_gaussian_varmin_fixed_mse_diffls_2_100.png)

#### With hyperparameters initialized to the same settings, but updated across iterations
![](sin_freq_1_noise_0.25_gaussian_varmin_estimated_mse_diffls_2_100.png)

---

### To improve the active learning strategy, we just need to estimate the hyperparameters. But do we need active learning in the first place?

Rather than choosing training points to maximize expected variance reduction, let's choose them randomly. This should perform worse; otherwise active learning is not worth it.

#### With random selection, the posterior after estimating hyperparameters achieves lower MSE faster than the active strategy
Although the active strategy might stabilize at a marginally lower MSE...

![](sin_freq_1_noise_0.25_gaussian_random_both_mse_2_100.png)

#### Posterior with random selection, estimating hyperparameters
![](sin_freq_1_noise_0.25_gaussian_random_estimated_posterior_1.00e+00_1.00e+00_2_100.png)

#### Distribution of randomly selected training points
![](sin_freq_1_noise_0.25_gaussian_random_estimated_training_density_1.00e+00_1.00e+00_2_100.png)

#### Path of hyperparameter estimates in log-marginal-likelihood landscape at final iteration
![](sin_freq_1_noise_0.25_gaussian_random_estimated_lml_1.00e+00_1.00e+00_2_100.png)

#### With different hyperparameter initializations that are updated across iterations, the random strategy achieves a similar or faster reduction in MSE
![](sin_freq_1_noise_0.25_gaussian_random_estimated_mse_diffls_2_100.png)

--- 

### What's going on? A pathological example
The active learning strategy may be choosing points that keep the estimated hyperparameters stuck in a local maximum of log likelihood. 

This is easier to see when we change the covariate space to uniform on [-1, 1].

### With random selection and estimated hyperparameters

#### Posterior
![](sin_freq_1_noise_0.25_uniform_random_estimated_posterior_1.00e+00_1.00e+00_2_500.png)

#### Path of hyperparameter estimates in log-marginal-likelihood landscape at final iteration
![](sin_freq_1_noise_0.25_uniform_random_estimated_lml_1.00e+00_1.00e+00_2_500.png)

### With active learning and estimated hyperparameters

This strategy ends up favoring the boundaries and continually estimates the wrong hyperparameters, in a self-reinforcing loop.

#### Posterior
![](sin_freq_1_noise_0.25_uniform_varmin_estimated_posterior_1.00e+00_1.00e+00_2_500.png)

#### Objective function computed by active learning algorithm
![](sin_freq_1_noise_0.25_uniform_varmin_estimated_objective_1.00e+00_1.00e+00_2_500.png)

#### Distribution of selected training points
![](sin_freq_1_noise_0.25_uniform_varmin_estimated_training_density_1.00e+00_1.00e+00_2_500.png)

#### Path of hyperparameter estimates in log-marginal-likelihood landscape at final iteration
![](sin_freq_1_noise_0.25_uniform_varmin_estimated_lml_1.00e+00_1.00e+00_2_500.png)

--- 

### Preliminary takeaways

- Estimating hyperparameters via evidence maximization results in a more accurate posterior than keeping them fixed.
- Based on MSE alone, it's not clear that GP active learning is preferable to random selection of training points.

#### TODO: evaluate posteriors on other metrics, like average predictive variance and KL divergence from ground truth

#### TODO: try for other covariate spaces, especially in higher dimensions
