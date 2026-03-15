########################################
##PreRegistration-SupermarktExperiment##
########################################


#In this file we present the code based on the description of our model and assumptions in the pre-registration.
#Based on this code the resepctive plots are generated.

# -----------------------------------------------------------------------
# Importing libraries and setting up
# -----------------------------------------------------------------------
#%%
import numpyro
import jax
print('Numpyro Version: ', numpyro.__version__)
print('Jax Version: ', jax.__version__)
import os
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import jax.numpy as jnp
from jax import random
import seaborn as sns
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

az.style.use("arviz-darkgrid")
numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)

rng_key = random.PRNGKey(1)

# -----------------------------------------------------------------------
# Defining the model
# -----------------------------------------------------------------------

#%%
# -----------------------------------------------------------------------
# LA-AIDS model
# Key features:
#   - Alpha sampled via Dirichlet (adds to 1 by construction, no w_bar needed)
#   - Beta fixed to zero (with fixed budget and only meat price varying, beta is collinear with meat price gammas, making it unidentified and attenuating gamma estimates)
#   - Only meat-row gammas sampled (only meat price varies, so only meat-row gammas are identified; fixing all others to zero is correct for this design)
# -----------------------------------------------------------------------
def aids_model_v3(treatment_idx, x_total=20.0, w_obs=None):
    N = treatment_idx.shape[0]
    prices = PRICES[treatment_idx]   # (N, 6)
    ln_p   = jnp.log(prices)        # (N, 6)

    # ================================================================
    # 1. ALPHA — sampled on the simplex via Dirichlet
    #    Doubles as Laspeyres price index weights (removes w_bar).
    #    Adding-up satisfied by construction.
    # ================================================================
    alpha = numpyro.sample(
        "alpha",
        dist.Dirichlet(jnp.ones(N_GOODS))
    )   # shape (6,), sums to 1

    # ================================================================
    # 2. BETA — FIXED TO ZERO
    #    Rationale: x_total is constant across all individuals, so
    #    ln(x/P*) = ln(20) - Σ alpha_i ln p_i is a deterministic
    #    function of ln_p_meat alone (only meat price varies).
    #    This makes beta collinear with gamma, attenuating both.
    #    Fixing beta=0 removes the collinearity entirely.
    # ================================================================
    beta = jnp.zeros(N_GOODS)
    # (no sample site — beta is not a parameter)

    # ================================================================
    # 3. GAMMA — only meat-row free parameters sampled
    #
    #    Identification argument:
    #      - Only ln_p_meat varies across treatments.
    #      - The 5 meat-row params (g00..g04) are identified by how
    #        each good's share responds to the meat price change.
    #      - All other gammas (g11..g44) are unidentified — fixing
    #        them to zero is the correct model for this design.
    #      - g05 derived via adding-up column sum = 0.
    #      - Last row derived by symmetry + adding-up.
    #
    #    Prior width: Normal(0, 0.50) — wide enough to cover plausible
    #    demand elasticities without being so wide that the sampler
    #    wastes time in implausible regions.
    # ================================================================
    g00 = numpyro.sample("g00", dist.Normal(0.0, 0.50))
    g01 = numpyro.sample("g01", dist.Normal(0.0, 0.50))
    g02 = numpyro.sample("g02", dist.Normal(0.0, 0.50))
    g03 = numpyro.sample("g03", dist.Normal(0.0, 0.50))
    g04 = numpyro.sample("g04", dist.Normal(0.0, 0.50))

    # Adding-up: column sums = 0
    # All non-meat cross-gammas are 0, so:
    #   col 1: g01 + 0 + ... = 0  =>  g15 = -g01, rest 0
    #   (same logic for cols 2–4)
    #   col 0: g00+g01+g02+g03+g04+g05 = 0
    g05 = -(g00 + g01 + g02 + g03 + g04)

    gamma = jnp.array([
        [ g00,  g01,  g02,  g03,  g04,  g05],   # meat row
        [ g01,  0.0,  0.0,  0.0,  0.0, -g01],   # meat-alt row
        [ g02,  0.0,  0.0,  0.0,  0.0, -g02],   # protein-alt row
        [ g03,  0.0,  0.0,  0.0,  0.0, -g03],   # other-pasta row
        [ g04,  0.0,  0.0,  0.0,  0.0, -g04],   # non-pasta row
        [ g05, -g01, -g02, -g03, -g04,  g01+g02+g03+g04-g05],  # change row
    ])
    numpyro.deterministic("gamma", gamma)

    # ================================================================
    # Laspeyres price index — alpha as weights (no separate w_bar)
    # ================================================================
    ln_P_star = jnp.sum(alpha[None, :] * ln_p, axis=-1, keepdims=True)  # (N,1)
    #becomes redundant as we set ß to 0, but we keep it here for clarity and potential future extensions

    # ================================================================
    # Linear predictor
    # beta = 0, so expenditure term drops out entirely
    # ================================================================
    lam = (
        alpha[None, :]       # (1, 6) baseline intercept
        + (ln_p @ gamma.T)   # (N, 6) price effects only
    )

    
    w_hat = jax.nn.softplus(lam)
    w_hat = w_hat / w_hat.sum(axis=-1, keepdims=True)
    numpyro.deterministic("w_hat", w_hat)

    # ================================================================
    # Likelihood
    # ================================================================
    concentration_scale = numpyro.sample(
        "concentration_scale",
        dist.HalfNormal(50.0)
    )

    numpyro.sample(
        "w_obs",
        dist.Dirichlet(w_hat * concentration_scale),
        obs=w_obs
    )

#%%
# -----------------------------------------------------------------------
# Generate synthetic data
# -----------------------------------------------------------------------


# Price configurations per treatment

P_BASELINE = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
P_VAT      = jnp.array([1.112, 1.0, 1.0, 1.0, 1.0, 1.0])
P_TAX_50   = jnp.array([1.5,   1.0, 1.0, 1.0, 1.0, 1.0])
P_TAX_100  = jnp.array([2.0,   1.0, 1.0, 1.0, 1.0, 1.0])

PRICES = jnp.stack([P_BASELINE, P_VAT, P_TAX_50, P_TAX_100])  # (4, 6)

#For now we assume no round effects, i.e. we have 400 observations, equally split across the 4 treatment groups (100 each).
#we consider 6 product groups: meat, meat alternatives, protein alternatives, other pasta, non-pasta, and change (the last category captures the difference between the sum of the first 5 categories and the total budget).
N_INDIVIDUALS   = 400
N_GOODS         = 6
N_GROUPS        = 4
N_OBS_PER_GROUP = N_INDIVIDUALS // N_GROUPS  # 100

#%%
# -----------------------------------------------------------------------
# True coefficients
#
# Key changes vs previous version:
#   1. beta removed (fixed to zero) — with a fixed budget and only meat
#      prices varying, ln(x/P*) is collinear with ln_p_meat, making beta
#      unidentified and attenuating gamma estimates.
#   2. g00/g01 kept within prior support — true values must be reachable
#      by the prior or the posterior is pulled toward zero by prior mass.
#      Rule of thumb: |true value| < 2 * prior_std to avoid heavy attenuation.
#   3. g02–g04 kept small — with a single price instrument these are only
#      weakly identified; large true values can't be recovered reliably.
# -----------------------------------------------------------------------
TRUE_COEFS = {
    # alpha: full simplex — sampled via Dirichlet, so condition on all 6
    "alpha": jnp.array([0.40, 0.15, 0.15, 0.05, 0.1, 0.1]), #set based on pre-test data

    # FIX 1: beta removed entirely — no beta in DGP
    # (beta_free site no longer exists in the model)

    # FIX 2: g00/g01 within prior support (prior std = 0.50)
    # Previous true values (-0.80, 0.50) were 8 and 5 prior SDs from zero
    "g00": -0.80,   # own-price meat       (was -0.80, prior sd now 0.50)
    "g01":  0.60,   # cross meat/meat-alt  (was  0.50, prior sd now 0.50)
    "g02":  0.40,   # cross meat/protein-alt

    # FIX 3: g03–g04 kept small — single instrument, weak identification    
    "g03":  0.06,   # cross meat/other-pasta
    "g04":  0.06,   # cross meat/non-pasta

    # Noise: lower concentration = more spread between treatments
    "concentration_scale": jnp.array(40.0),
}

# -----------------------------------------------------------------------
# Condition on TRUE_COEFS (sampled sites only: alpha, g00-g04,
# concentration_scale). No beta site exists in v3.
# -----------------------------------------------------------------------
condition_model = numpyro.handlers.condition(aids_model_v3, data=TRUE_COEFS)
prior_predictive = Predictive(condition_model, num_samples=1)

rng_key, rng_key_ = random.split(rng_key)
treatment_idx_all = jnp.repeat(jnp.arange(N_GROUPS), N_OBS_PER_GROUP)

prior_preds = prior_predictive(rng_key_, treatment_idx=treatment_idx_all)

print("Alpha from TRUE_COEFS:", np.array(prior_preds["alpha"]))
print("Synthetic data shape:", prior_preds["w_obs"].shape)

synthetic_data = {
    "treatment_idx": treatment_idx_all,
    "w_obs": prior_preds["w_obs"].reshape(-1, N_GOODS),
}

#%%
# -----------------------------------------------------------------------
# Prior predictive check
# -----------------------------------------------------------------------
w_obs_arr    = np.array(synthetic_data["w_obs"])
treatment_arr = np.array(treatment_idx_all)
group_names  = {0: "Baseline", 1: "VAT 7→19%", 2: "Meat Tax 50%", 3: "Meat Tax 100%"}

df_prior = pd.DataFrame({
    "meat_share":        w_obs_arr[:, 0],
    "meat_alt_share":    w_obs_arr[:, 1],
    "protein_alt_share": w_obs_arr[:, 2],
    "treatment": [group_names[t] for t in treatment_arr],
})

goods  = ["meat_share", "meat_alt_share", "protein_alt_share"]
titles = ["Meat", "Meat Alternatives", "Protein Alternatives"]

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
for ax, good, title in zip(axes, goods, titles):
    sns.kdeplot(data=df_prior, x=good, hue="treatment", fill=True, alpha=0.4, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Budget Share")
    ax.set_xlim(0, 1)


handles, labels = axes[0].get_legend().legend_handles, [t.get_text() for t in axes[0].get_legend().get_texts()]


for ax in axes:
    if ax.get_legend():
        ax.get_legend().remove()


fig.legend(
    handles, labels,
    title="Treatment",
    loc="lower center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=4,
    frameon=True,
)
plt.suptitle("Synthetic Data: Budget Shares by Treatment", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/Figure1_prior_predictive_check.png", dpi=150)
plt.show()

print("\nMean shares per treatment (synthetic data):")
print(df_prior.groupby("treatment")[goods].mean().round(3))

#%%
# -----------------------------------------------------------------------
# Run MCMC
# -----------------------------------------------------------------------
mcmc = MCMC(
    NUTS(aids_model_v3),
    num_warmup=1000,
    num_samples=1000,
    num_chains=4,
)
rng_key, rng_key_ = random.split(rng_key)
mcmc.run(rng_key_, **synthetic_data)
mcmc.print_summary()

#%%
# -----------------------------------------------------------------------
# Posterior predictive
# -----------------------------------------------------------------------
posterior_samples = mcmc.get_samples()
predictive = Predictive(aids_model_v3, posterior_samples)
rng_key, rng_key_ = random.split(rng_key)
post_preds = predictive(rng_key_, treatment_idx=treatment_idx_all)

n_post_samples = posterior_samples[list(posterior_samples.keys())[0]].shape[0]

#%%
# -----------------------------------------------------------------------
# Prior vs posterior predictive — meat share
# -----------------------------------------------------------------------
rng_key, rng_key_ = random.split(rng_key)
prior_pred_free = Predictive(aids_model_v3, num_samples=n_post_samples)
prior_preds_free = prior_pred_free(rng_key_, treatment_idx=treatment_idx_all)

w_obs_prior_meat = np.array(prior_preds_free["w_obs"][:, :, 0].reshape(-1))
w_obs_post_meat  = np.array(post_preds["w_obs"][:, :, 0].reshape(-1))

plt.figure(figsize=(8, 5))
sns.kdeplot(x=w_obs_prior_meat, fill=True, color="salmon",  alpha=0.5, label="Prior predictive")
sns.kdeplot(x=w_obs_post_meat,  fill=True, color="skyblue", alpha=0.5, label="Posterior predictive")
plt.title("Prior vs Posterior Predictive — Meat Share", fontsize=14)
plt.xlabel("Meat Share")
plt.ylabel("Density")
plt.xlim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig("outputs/Figure2_prior_vs_posterior_meat.png", dpi=150)
plt.show()

#%%
# -----------------------------------------------------------------------
# Parameter recovery — gamma (meat row)
# -----------------------------------------------------------------------
gamma_elements = [
    (0, 0, TRUE_COEFS["g00"], "g00 own-price meat"),
    (0, 1, TRUE_COEFS["g01"], "g01 meat/meat-alt"),
    (0, 2, TRUE_COEFS["g02"], "g02 meat/protein-alt"),
    (0, 3, TRUE_COEFS["g03"], "g03 meat/other-pasta"),
    (0, 4, TRUE_COEFS["g04"], "g04 meat/non-pasta"),
]

gamma_post  = np.array(posterior_samples["gamma"])
gamma_prior = np.array(prior_preds_free["gamma"])

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes_flat = axes.flatten()  

for ax, (row, col, true_val, label) in zip(axes_flat, gamma_elements):
    true_val = float(true_val)
    post_vals  = gamma_post[:, row, col]
    prior_vals = gamma_prior[:, row, col]
    sns.kdeplot(post_vals,  ax=ax, fill=True, color="steelblue", alpha=0.5, label="Posterior")
    sns.kdeplot(prior_vals, ax=ax, fill=True, color="green",     alpha=0.3, label="Prior")
    ax.axvline(true_val, color="red", linestyle="--", label=f"True: {true_val:.3f}")
    hdi = az.hdi(post_vals, hdi_prob=0.94)
    ax.axvline(hdi[0], color="steelblue", linestyle=":", linewidth=1.5)
    ax.axvline(hdi[1], color="steelblue", linestyle=":", linewidth=1.5)
    ax.set_title(f"γ {label}\nTrue={true_val:.3f}\nHPDI=[{hdi[0]:.3f}, {hdi[1]:.3f}]")
    ax.set_xlabel("Value")

axes_flat[-1].set_visible(False)  
axes_flat[0].legend()
plt.suptitle("Prior vs Posterior: γ (meat row — identifiable)", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/Figure3_gamma_recovery.png", dpi=150)
plt.show()

#%%
# -----------------------------------------------------------------------
# Parameter recovery — alpha
# -----------------------------------------------------------------------
alpha_true = np.array(TRUE_COEFS["alpha"])
alpha_labels = ["Meat", "Meat Alt", "Protein Alt", "Other Pasta", "Non-Pasta", "Change"]
alpha_post  = np.array(posterior_samples["alpha"])
alpha_prior = np.array(prior_preds_free["alpha"])

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, (ax, label) in enumerate(zip(axes, alpha_labels)):
    sns.kdeplot(alpha_post[:, i],  ax=ax, fill=True, color="steelblue", alpha=0.5, label="Posterior")
    sns.kdeplot(alpha_prior[:, i], ax=ax, fill=True, color="green",     alpha=0.3, label="Prior")
    ax.axvline(alpha_true[i], color="red", linestyle="--", label=f"True: {alpha_true[i]:.3f}")
    hdi = az.hdi(alpha_post[:, i], hdi_prob=0.94)
    ax.axvline(hdi[0], color="steelblue", linestyle=":", linewidth=1.5)
    ax.axvline(hdi[1], color="steelblue", linestyle=":", linewidth=1.5)
    ax.set_title(f"α {label}\nTrue={alpha_true[i]:.3f}, HPDI=[{hdi[0]:.3f}, {hdi[1]:.3f}]")
    ax.set_xlabel("Value")
axes[0].legend()
plt.suptitle("Prior vs Posterior: α (baseline shares)", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/Figure4_alpha_recovery.png", dpi=150)
plt.show()

#%%
# -----------------------------------------------------------------------
# Posterior predictive by treatment
# -----------------------------------------------------------------------
treatment_tiled = np.tile(
    np.repeat(np.arange(N_GROUPS), N_OBS_PER_GROUP),
    n_post_samples
)
w_obs_post = np.array(post_preds["w_obs"].reshape(-1, N_GOODS))

df_post = pd.DataFrame({
    "meat_share":        w_obs_post[:, 0],
    "meat_alt_share":    w_obs_post[:, 1],
    "protein_alt_share": w_obs_post[:, 2],
    "treatment": [group_names[t] for t in treatment_tiled],
})

# missing: create figure and plot first
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
for ax, good, title in zip(axes, goods, titles):
    sns.kdeplot(data=df_post, x=good, hue="treatment", fill=True, alpha=0.4, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Budget Share")
    ax.set_xlim(0, 1)

# now grab handles before removing legends
handles, labels = axes[0].get_legend().legend_handles, [t.get_text() for t in axes[0].get_legend().get_texts()]

for ax in axes:
    if ax.get_legend():
        ax.get_legend().remove()

fig.legend(
    handles, labels,
    title="Treatment",
    loc="lower center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=4,
    frameon=True,
)

plt.suptitle("Posterior Predictive Budget Shares by Treatment", fontsize=13)
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("outputs/Figure5_posterior_predictive_by_treatment.png", dpi=150, bbox_inches="tight")
plt.show()


print("\nPosterior predictive mean shares per treatment:")
print(df_post.groupby("treatment")[goods].mean().round(3))

# %%
