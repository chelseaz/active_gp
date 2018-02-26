# Comparing random and active selection in two dimensions

The true response surface is the paraboloid $z = x^2 + y^2$ with independent Gaussian noise of variance 0.25.

The GP for estimating the response surface consists of an RBF kernel with additive white noise. Hyperparameters are estimated after every additional training point by maximizing the marginal likelihood of the GP. Initial hyperparameters are `length_scale=1.0, noise_level=1.0`.

Training points are normally distributed with mean zero and identity covariance.

To choose the next point, the active learning algorithm computes the expected reduction in predictive variance for a candidate training point by Monte Carlo, averaging over 500 query points. This objective is evaluated for 300 candidate training points. All query points and candidate training points are independently drawn from the covariate distribution. The training point with greatest expected variance reduction is chosen, and we observe its output value.

Random selection | Predictive variance minimization
--- | ---
![](paraboloid_1_noise_0.25_mvgaussian_random_estimated_posterior_1.00e+00_1.00e+00_11_500.gif) | ![](paraboloid_1_noise_0.25_mvgaussian_varmin_estimated_posterior_1.00e+00_1.00e+00_11_500.gif)
![](paraboloid_1_noise_0.25_mvgaussian_random_estimated_lml_1.00e+00_1.00e+00_11_500.gif) | ![](paraboloid_1_noise_0.25_mvgaussian_varmin_estimated_lml_1.00e+00_1.00e+00_11_500.gif)
![](paraboloid_1_noise_0.25_mvgaussian_random_estimated_training_density_1.00e+00_1.00e+00_11_500.gif) | ![](paraboloid_1_noise_0.25_mvgaussian_varmin_estimated_training_density_1.00e+00_1.00e+00_11_500.gif)
 | ![](paraboloid_1_noise_0.25_mvgaussian_varmin_estimated_objective_1.00e+00_1.00e+00_11_500.gif)

### Observations

- Random selection learns the hyperparameters faster. Active selection vacillates at first between local maxima of log marginal likelihood.
- Once active selection arrives at reasonable hyperparameters, it learns a better posterior mean. Hence active selection reduces MSE faster in later iterations:

Strategy | Metrics
--- | ---
**Random selection** | ![](paraboloid_1_noise_0.25_mvgaussian_random_estimated_eval_1.00e+00_1.00e+00_11_500.png)
**Predictive variance minimization** | ![](paraboloid_1_noise_0.25_mvgaussian_varmin_estimated_eval_1.00e+00_1.00e+00_11_500.png)

Note: Expected posterior predictive variance depends on hyperparameters, so the right side plots are not comparable. Need to compute something like KL between ground truth and posterior.