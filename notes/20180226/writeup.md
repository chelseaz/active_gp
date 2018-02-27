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

### With different length-scale initializations

Initially `noise_level=0.25`, the true noise variance.

Strategy | Metrics
--- | ---
**Random selection** | ![](paraboloid_1_noise_0.25_mvgaussian_random_estimated_eval_diffls_11_500.png)
**Predictive variance minimization** | ![](paraboloid_1_noise_0.25_mvgaussian_varmin_estimated_eval_diffls_11_500.png)

### With different variance initializations

Initially `length_scale=1.0`, close to the true length-scale.

Strategy | Metrics
--- | ---
**Random selection** | ![](paraboloid_1_noise_0.25_mvgaussian_random_estimated_eval_diffvar_11_500.png)
**Predictive variance minimization** | ![](paraboloid_1_noise_0.25_mvgaussian_varmin_estimated_eval_diffvar_11_500.png)

Note: MSE is computed over the randomness of covariate space, for one run of the algorithm. Need to compute MSE of posterior mean at any point in covariate space by averaging over runs (over randomness in the training set).

# In higher dimensions

The true response surface is a d-dimensional paraboloid, $y = 0.1 \sum_{i=1}^d x_i^2$. Initial hyperparameters are `length_scale=1.0, noise_level=1.0`.

## d = 5

### Random selection strategy

Metrics | Hyperparameter path
--- | ---
![](5d_paraboloid_0.2_noise_0.25_mvgaussian_random_estimated_eval_1.00e+00_1.00e+00_11_1000.png) | ![](5d_paraboloid_0.2_noise_0.25_mvgaussian_random_estimated_lml_1.00e+00_1.00e+00_11_1000.gif)

### Predictive variance minimization

Metrics | Hyperparameter path
--- | ---
![](5d_paraboloid_0.2_noise_0.25_mvgaussian_varmin_estimated_eval_1.00e+00_1.00e+00_11_1000.png) | ![](5d_paraboloid_0.2_noise_0.25_mvgaussian_varmin_estimated_lml_1.00e+00_1.00e+00_11_1000.gif)

## d = 10

### Random selection strategy

Metrics | Hyperparameter path
--- | ---
![](10d_paraboloid_0.1_noise_0.25_mvgaussian_random_estimated_eval_1.00e+00_1.00e+00_11_1000.png) | ![](10d_paraboloid_0.1_noise_0.25_mvgaussian_random_estimated_lml_1.00e+00_1.00e+00_11_1000.gif)

### Predictive variance minimization

Metrics | Hyperparameter path
--- | ---
![](10d_paraboloid_0.1_noise_0.25_mvgaussian_varmin_estimated_eval_1.00e+00_1.00e+00_11_1000.png) | ![](10d_paraboloid_0.1_noise_0.25_mvgaussian_varmin_estimated_lml_1.00e+00_1.00e+00_11_1000.gif)

### Observations

- In 5 dimensions, random and active selection perform comparably. Active selection may reach a slightly lower MSE, but random selection achieves a better estimate of the noise variance.
- In higher dimensions (d=10), the active strategy fails to reach reasonable hyperparameter estimates, and its MSE grows worse over 1000 training points. The random strategy stabilizes at nearly correct hyperparameters by about 200 points, and its MSE declines accordingly.