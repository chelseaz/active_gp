# Sequentially estimating hyperparameters with strategy of minimizing predictive variance
./sequential_gp.py --mode seq --strategy varmin --covariate-space gaussian --ground-truth low_freq_sinusoid --nmin 2 --nmax 100

# Compare to fixed hyperparameters
./sequential_gp.py --mode compare-seq-fixed --strategy varmin --covariate-space gaussian --ground-truth low_freq_sinusoid --nmin 2 --nmax 100

# Compare estimated to fixed hyperparameters, selecting training points randomly
./sequential_gp.py --mode compare-seq-fixed --strategy random --covariate-space gaussian --ground-truth low_freq_sinusoid --nmin 2 --nmax 100

# Sequentially estimating hyperparameters with strategy of minimizing predictive variance, comparing different hyperparameter initializations
./sequential_gp.py --mode compare-seq-seq --strategy varmin --covariate-space gaussian --ground-truth low_freq_sinusoid --nmin 2 --nmax 100

# How badly does the variance-minimizing strategy do under different fixed hyperparameter settings?
./sequential_gp.py --mode compare-fixed-fixed --strategy varmin --covariate-space gaussian --ground-truth low_freq_sinusoid --nmin 2 --nmax 100

# Repeat the above analysis, selecting training points randomly instead of via the variance-minimizing strategy.