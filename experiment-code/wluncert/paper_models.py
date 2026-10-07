"""Bayesian models exactly as written in Eqs. 1-3 of the HyPerf paper (ICSE 2026).

Reading of the notation:
- N(m, s^2) and every quantity written as a square (sigma^2) is a variance;
  the code samples the variance and uses its square root as the scale.
- Exp(x) is an exponential distribution with mean x (NumPyro: rate 1/x);
  for Exp(1) the rate and the mean reading coincide.
- L(0, b) is a Laplace distribution with scale b.

Non-centered parameterizations (LocScaleReparam) do not change the model;
they only help the NUTS sampler.
"""

import numpy as np
import numpyro
import numpyro.distributions as npdist
from jax import numpy as jnp
from numpyro.infer.reparam import LocScaleReparam

from models import (
    MCMCCombinedNoPooling,
    MCMCCombinedCompletePooling,
    MCMCPartialRobustLassoAdaptiveShrinkage,
)


def _observe(mean, scale, reference_y):
    with numpyro.plate("data", mean.shape[0]):
        return numpyro.sample(
            "observations", npdist.Normal(mean, scale), obs=reference_y
        )


class PaperNoPooling(MCMCCombinedNoPooling):
    """Eq. 1: alpha_s ~ N(0,1), b ~ Exp(1), beta_s ~ L(0,b), sigma_s^2 ~ Exp(1)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_id = "mcmc-paper-eq1-no-pooling"

    def get_reparam_dict(self):
        return {"influences": LocScaleReparam(0), "base": LocScaleReparam(0)}

    def model(self, data, workloads, n_workloads, reference_y):
        num_opts = data.shape[1]
        b = numpyro.sample("influences-regularization-shrinkage", npdist.Exponential(1.0))
        with numpyro.plate("options", num_opts):
            with numpyro.plate("workloads", n_workloads):
                influences = numpyro.sample("influences", npdist.Laplace(0.0, b))
        with numpyro.plate("workloads", n_workloads):
            base = numpyro.sample("base", npdist.Normal(0.0, 1.0))
            error_var = numpyro.sample("error-var", npdist.Exponential(1.0))
            error = numpyro.deterministic("error", jnp.sqrt(error_var))
        mean = jnp.multiply(data, influences[workloads]).sum(axis=1).ravel()
        mean = mean + base[workloads]
        return _observe(mean, error[workloads], reference_y)


class PaperCompletePooling(MCMCCombinedCompletePooling):
    """Eq. 2: alpha ~ N(0,1), b ~ Exp(1), beta ~ L(0,b), sigma^2 ~ Exp(1)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_id = "mcmc-paper-eq2-complete-pooling"

    def get_reparam_dict(self):
        return {"influences": LocScaleReparam(0), "base": LocScaleReparam(0)}

    def model(self, data, workloads, n_workloads, reference_y):
        num_opts = data.shape[1]
        b = numpyro.sample("influences-regularization-shrinkage", npdist.Exponential(1.0))
        with numpyro.plate("options", num_opts):
            influences = numpyro.sample("influences", npdist.Laplace(0.0, b))
        base = numpyro.sample("base", npdist.Normal(0.0, 1.0))
        error_var = numpyro.sample("error-var", npdist.Exponential(1.0))
        error = numpyro.deterministic("error", jnp.sqrt(error_var))
        mean = jnp.multiply(data, influences).sum(axis=1).ravel() + base
        return _observe(mean, error, reference_y)


class PaperPartialPooling(MCMCPartialRobustLassoAdaptiveShrinkage):
    """Eq. 3 (HyPerf):
    alpha_mu ~ N(0,1), alpha_sigma^2 ~ Exp(1), alpha_s ~ N(alpha_mu, alpha_sigma^2),
    b ~ Exp(1), beta_mu ~ L(0,b), beta_sigma^2 ~ Exp(1), beta_s ~ N(beta_mu, beta_sigma^2),
    sigma_mu^2 ~ Exp(1), sigma_s^2 ~ Exp(sigma_mu^2) (mean sigma_mu^2).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_id = "mcmc-paper-eq3-partial-pooling"

    def get_reparam_dict(self):
        return {
            "influences": LocScaleReparam(0),
            "base": LocScaleReparam(0),
            "influences-mean-hyperior": LocScaleReparam(0),
        }

    def model(self, data, workloads, n_workloads, reference_y):
        num_opts = data.shape[1]
        # intercepts (Eqs. 3a, 3e)
        base_mu = numpyro.sample("base-hyper", npdist.Normal(0.0, 1.0))
        base_var = numpyro.sample("base-hyper-var", npdist.Exponential(1.0))
        # option influences (Eqs. 3c, 3d, 3f)
        b = numpyro.sample("influences-regularization-shrinkage", npdist.Exponential(1.0))
        with numpyro.plate("options", num_opts):
            beta_mu = numpyro.sample("influences-mean-hyperior", npdist.Laplace(0.0, b))
            beta_var = numpyro.sample("influences-var-hyperior", npdist.Exponential(1.0))
            beta_sd = numpyro.deterministic("influences-stddevs-hyperior", jnp.sqrt(beta_var))
        with numpyro.plate("options", num_opts):
            with numpyro.plate("workloads", n_workloads):
                influences = numpyro.sample("influences", npdist.Normal(beta_mu, beta_sd))
        # noise (Eq. 3b): sigma_mu^2 ~ Exp(1), sigma_s^2 ~ Exp with mean sigma_mu^2
        error_hyper_var = numpyro.sample("error-hyper-var", npdist.Exponential(1.0))
        with numpyro.plate("workloads", n_workloads):
            base = numpyro.sample("base", npdist.Normal(base_mu, jnp.sqrt(base_var)))
            error_var = numpyro.sample("error-var", npdist.Exponential(1.0 / error_hyper_var))
            error = numpyro.deterministic("error", jnp.sqrt(error_var))
        mean = jnp.multiply(data, influences[workloads]).sum(axis=1).ravel()
        mean = mean + base[workloads]
        return _observe(mean, error[workloads], reference_y)


PAPER_MODEL_CLASSES = {
    "paper-no-pooling-mcmc": PaperNoPooling,
    "paper-cpooling-mcmc": PaperCompletePooling,
    "paper-partial-pooling-mcmc": PaperPartialPooling,
}
