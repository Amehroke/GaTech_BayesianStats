import pandas as pd
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az


# Re-load the dataset
file_path = "../Data/Hw5/enzyme-2.csv"
data = pd.read_csv(file_path)

# Convert data columns to NumPy arrays
x_values = data["x"].values
y_values = data["y"].values

# Redefine the Bayesian nonlinear regression model using PyMC
with pm.Model() as model_fixed:
    # Priors for theta_1 and theta_2 (positive-only)
    theta_1 = pm.HalfNormal("theta_1", sigma=100)
    theta_2 = pm.HalfNormal("theta_2", sigma=1)
    
    # Prior for standard deviation (positive-only)
    sigma = pm.HalfNormal("sigma", sigma=50)
    
    # Nonlinear regression function
    mu = (theta_1 * x_values) / (theta_2 + x_values)
    
    # Likelihood
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_values)
    
    # Sample from the posterior using MCMC
    trace_fixed = pm.sample(2000, tune=1000, return_inferencedata=True, target_accept=0.9)

# Plot marginal posterior densities
az.plot_posterior(trace_fixed, var_names=["theta_1", "theta_2", "sigma"])
plt.show()
