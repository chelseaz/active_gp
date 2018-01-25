# Non-sequential with fixed hyperparameters
./sequential_gp.py --fix-theta --covariate-space centered --ground-truth low_freq_sinusoid --nmin 11 --nmax 1000 

# Sequential, estimating hyperparameters
./sequential_gp.py --covariate-space centered --ground-truth low_freq_sinusoid --nmin 11 --nmax 1000 