from __future__ import print_function
import os
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as pl
import emcee
import time
import shutil
import configparser  # Import the configparser module

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

print(f"Parameters extracted:\n npoints: {npoints}\n nwalkers: {nwalkers}\n burnin_fraction: {burnin_fraction}\n output_dir: {output_dir}\n save_steps: {save_steps}\n ndim: {ndim}\n p0: {p0}")

# Reproducible results!
np.random.seed(123)

def Hz1(params, zz):
    h0, k, q = params
    model = (1/((1+k)**0.5))*100.0*h0*(((1.0+zz)**(0.25*(1.0+3.0*q-(1.0 + 6.0*q + 9.0*q**2.0 + 36.0*(q-(2/3)))**0.5))))*(k + (1.0 + zz)**((1.0 + 6.0*q + 9.0*q**2.0 + 36.0*(q-(2/3)))**0.5))**0.5
    return model

# Observational Hubble data
data = np.loadtxt("ohd_data.dat")
z = data[:,0]
H = data[:,1]
err = data[:,2]

def chiohd(params, z, H, err):
    h0, k, q = params
    return np.sum(((H - Hz1(params, z))/err)**2.)

# BAO data:
c = 3.0*10**5
oc = 9.2*10**-5.

def Hz(params, zz):
    h0, k, q = params
    model = (1/((1+k)**0.5))*100.0*h0*(((1.0+zz)**(0.25*(1.0+3.0*q-(1.0 + 6.0*q + 9.0*q**2.0 + 36.0*(q-(2/3)))**0.5))))*(k + (1.0 + zz)**((1.0 + 6.0*q + 9.0*q**2.0 + 36.0*(q-(2/3)))**0.5))**0.5
    return model

def dA(params, zz):
    h0, k, q = params
    (A2, A1) = integrate.quad(lambda x: c/Hz(params, x), 0., zz)
    return abs(A2 - A1)

def Dv(params, zz):
    h0, k, q = params
    return ((dA(params, zz)**2.)*c*zz/Hz(params, zz))**(1./3.)

def X(params):
    h0, k, q = params
    return np.array([[dA(params,1091)/Dv(params,0.106) - 30.43],
                     [dA(params,1091)/Dv(params,0.32) - 11.0],
                     [dA(params,1091)/Dv(params,0.57) - 6.77]], float)

M = np.array([[0.2029, 0.0, 0.0], [0.0, 7.3046, 0.0], [0.0, 0.0, 39.0625]], float)

def chibao(params):
    h0, k, q = params
    return np.dot(np.transpose(X(params)), np.dot(M, X(params)))

# JLA data:
data_mu = np.loadtxt('jla_mub_0.txt')
z_tab_sne = data_mu[:,0]
mu_tab_sne = data_mu[:,1]
data_cov = np.loadtxt('jla_mub_covmatrix.txt')
data_cov_sne = np.reshape(data_cov, (len(mu_tab_sne), len(mu_tab_sne)))
err_mu_sne = np.sqrt(np.diagonal(data_cov_sne))
Inv_cov_sne = np.linalg.inv(data_cov_sne)

def dL(params, zz):
    h0, k, q = params
    (dl2, dl1) = integrate.quad(lambda x: (1.0/Hz1(params, x)), 0., zz)
    return abs(c*(1.+zz)*(dl2-dl1))

def mu_model(params, zz):
    h0, k, q = params
    return 5.0*(np.log10([dL(params, zd) for zd in zz])) + 25.0

def chisne(params, z_tab_sne, mu_tab_sne, data_cov_sne):
    h0, k, q = params
    Dmu = np.subtract(mu_tab_sne, mu_model(params, z_tab_sne))
    return np.dot(Dmu, np.dot(Inv_cov_sne, Dmu))

def chiohdbaosne(params, z, H, err):
    h0, k, q = params
    return chiohd(params, z, H, err) + chisne(params, z_tab_sne, mu_tab_sne, data_cov_sne) + chibao(params)

def lnlike(params, z, H, err):
    try:
        h0, k, q = params
        return -chiohdbaosne(params, z, H, err) * 0.5
    except Exception as e:
        print("Error in lnlike:", e)
        return -np.inf

def lnprior(params):
    h0, k, q = params
    if h0_min < h0 < h0_max and k_min < k < k_max and q_min < q < q_max:
        return 0.0
    return -np.inf

def lnprob(params, z, H, err):
    lp = lnprior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(params, z, H, err)


# Set up the sampler parameters
pos = [p0 + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

# Initialize the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(z, H, err))

# Initialize arrays to store MCMC chain and likelihoods
chain_arr = np.empty([nwalkers, npoints, ndim])
lnprob_arr = -np.inf * np.ones([nwalkers, npoints])

# Remove previous data if it exists
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# Create the directory for new samples
os.makedirs(output_dir)

# Run the MCMC sampler
start_time = time.time()
#for step_index, (pos, lnprob, state) in enumerate(sampler.sample(pos, iterations=npoints, progress=True)):
for step_index, (pos, lnprob, state) in enumerate(sampler.sample(pos, iterations=npoints, progress=True)):
    print(f"Step: {step_index + 1}/{npoints}")  # Print current step

    # Check if step_index exceeds array bounds
    if step_index >= npoints:
        break

    chain_arr[:, step_index, :] = pos
    lnprob_arr[:, step_index] = lnprob

    if (np.remainder(step_index + 1, save_steps) == 0):
        pos_tmp = np.reshape(chain_arr, (nwalkers * npoints, ndim))
        lnprob_tmp = np.reshape(lnprob_arr, (nwalkers * npoints))

        like_max = np.max(lnprob_tmp)
        best = pos_tmp[np.argmax(lnprob_tmp), :]

        print('\ntime (m), steps, best-fit param, max likelihood, mean acceptance fraction: ', (time.time() - start_time) / 60., step_index + 1, best, like_max, np.mean(sampler.acceptance_fraction))

        for k in range(nwalkers):
            file_path = os.path.join(output_dir, f"walker_{k+1}.txt")
            with open(file_path, "a") as f:
                for i in range(step_index - save_steps + 1, step_index + 1):
                    s = "{0:6d}".format(1)  # Dummy for weight
                    s += " " + "{:.6e}".format(lnprob_arr[k, i])
                    for kk in range(ndim):
                        s += " " + "{:.6e}".format(chain_arr[k, i, kk])
                    s += "\n"
                    f.write(s)

print(f"MCMC run time (in min): {(time.time() - start_time) / 60.0}")

# Apply burn-in
burnin_steps = int(burnin_fraction * npoints)
samples_after_burnin = chain_arr[:, burnin_steps:, :].reshape(-1, ndim)
lnprob_after_burnin = lnprob_arr[:, burnin_steps:].reshape(-1)

# Find the best-fit point
max_lnprob_index = np.argmax(lnprob_after_burnin)  # Index of the maximum likelihood
best_fit_params = samples_after_burnin[max_lnprob_index]  # Parameters corresponding to the max likelihood

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

print("MCMC run finished ...")

