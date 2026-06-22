import os
import numpy as np
import configparser  # Import the configparser module
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from getdist import plots, MCSamples

# Read configuration from the .ini file
config = configparser.ConfigParser()
config.read('config.ini')

# Extract parameters from the .ini file
npoints = config.getint('DEFAULT', 'npoints')
nwalkers = config.getint('DEFAULT', 'nwalkers')
burnin_fraction = config.getfloat('DEFAULT', 'burnin_fraction')
output_dir = config.get('DEFAULT', 'output_dir')
save_steps = config.getint('DEFAULT', 'save_steps')
ndim = config.getint('DEFAULT', 'ndim')

# Extract starting point p0
p0_str = config.get('DEFAULT', 'p0').strip()
try:
    p0 = np.array([float(x) for x in p0_str.split(',')])
except ValueError as e:
    print(f"Error parsing p0: {e}")
    raise

# Extract parameter ranges
h0_min = config.getfloat('PARAMETER_RANGES', 'h0_min')
h0_max = config.getfloat('PARAMETER_RANGES', 'h0_max')
k_min = config.getfloat('PARAMETER_RANGES', 'k_min')
k_max = config.getfloat('PARAMETER_RANGES', 'k_max')
q_min = config.getfloat('PARAMETER_RANGES', 'q_min')
q_max = config.getfloat('PARAMETER_RANGES', 'q_max')


# Initialize lists to store samples and log-probabilities
samples = []
lnprobs = []
burnin_steps = int(burnin_fraction * npoints)  # Burn-in steps per walker
print("Loading MCMC data...")

# Load all walker data and apply burn-in to each walker separately
for k in range(nwalkers):
    file_path = os.path.join(output_dir, f"walker_{k+1}.txt")
    walker_data = np.loadtxt(file_path)

    # Apply burn-in
    walker_samples = walker_data[burnin_steps:, 2:2 + ndim]  # Extract samples after burn-in
    walker_lnprobs = walker_data[burnin_steps:, 1]           # Extract lnprob after burn-in

    # Append post-burn-in samples to the lists
    samples.append(walker_samples)
    lnprobs.append(walker_lnprobs)

# Combine all post-burn-in samples and lnprobs into single arrays
samples_after_burnin = np.vstack(samples)
lnprob_after_burnin = np.hstack(lnprobs)

# Check consistency of loaded data
print(f"Loaded {samples_after_burnin.shape[0]} post-burn-in samples with {samples_after_burnin.shape[1]} dimensions.")
print(f"Log-probabilities range: {lnprob_after_burnin.min()} to {lnprob_after_burnin.max()}")

# Find the best-fit point
max_lnprob_index = np.argmax(lnprob_after_burnin)
best_fit_params = samples_after_burnin[max_lnprob_index]

print("Best-fit parameters:")
print(f"h0: {best_fit_params[0]:.3f}, k: {best_fit_params[1]:.3f}, q: {best_fit_params[2]:.3f}")

# Calculate credible intervals
h0_mcmc, k_mcmc, q_mcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                              zip(*np.percentile(samples_after_burnin, [16, 50, 84], axis=0)))
print("""MCMC result:
   h0: {:.3f} +{:.3f} -{:.3f}
   k: {:.3f} +{:.3f} -{:.3f}
   q: {:.3f} +{:.3f} -{:.3f}""".format(h0_mcmc[0], h0_mcmc[1], h0_mcmc[2],
                                       k_mcmc[0], k_mcmc[1], k_mcmc[2],
                                       q_mcmc[0], q_mcmc[1], q_mcmc[2]))

print("MCMC analysis finished ...")

print("getdist plot")
truth_arr = best_fit_params

   #########  CHANGE HERE  #########################################
names = ['h0','k','q']
labels =  [r'h_{0}',r'C_{1}',r'\delta']
samples = MCSamples(samples=samples_after_burnin,names = names, labels = labels, settings={'smooth_scale_2D':0.3})
   
########################################################################
g = plots.get_subplot_plotter(width_inch=6)
samples.updateSettings({'contours':[0.68,0.95,0.99]})
g.settings.num_plot_contours = 3
g.triangle_plot(samples, names, filled=True, contour_colors=['purple']) # lims=[0, 0.0022, 0, 0.0022])
# Save the plot
g.export('hubble.png')

########################################################################

