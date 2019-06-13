
# Diversity
Tools for measuring heterogeneity

## Installation

```
pip install git+https://github.com/abrahamnunes/diversity.git
```

## Documentation

Documentation is provided in the Jupyter notebooks located in the `docs/` directory.

## Measures implemented

- Berger-Parker Diversity index: `berger_parker_diversity`
- Berger-Parker Dominance index: `berger_parker_dominance`
- Rényi heterogeneity (aka. Hill numbers): `renyi`
    - Includes decomposition (`renyi_decomp`) into $\\alpha$ (`renyi_alpha`) and $\\beta$ components
- Rényi entropy: `renyi_entropy`
- Coefficient of variation: `coef_variation`
- Freeman index: `freeman_index`
- Generalized Entropy Index: `gei`
- Gini-Simpson index: `gini_simpson`
- Hartley heterogeneity: `hartley`
- Hartley entropy: `hartley_entropy`
- Lincoln index (base, Chapman, and Bayesian estimators): `lincoln_index`
- Mean Log-Deviation: `mean_logdev`
- ModVR (Variation around the Mode): `ModVR`
- Range (simple): `range`
- RanVR (Range around the Mode): `RanVR`
- Shannon heterogeneity: `shannon`
- Shannon entropy: `shannon_entropy`
- Simpson index (aka Herfindahl index): `simpson`
- Simpson dominance: `simpson_dominance`
- Theil index: `theil`
- Tsallis entropy: `tsallis_entropy`

## Methods for Estimation

- Bootstrap sampling: `bootci`
    - State of the art approach is implemented: bias-corrected and accelerated [BCa] confidence intervals

## Measures to Add

- Accumulation curves: `accumulation_curve`
- Amato's index: `amato`
- Atkinson index: `atkinson`
- Average deviation: `AvDev`
- Birthday paradox estimate: `bday_paradox_test`
- Brillouin's D: `brillouin_d`
- Brillouin's E: `brillouin_e`
- Chao & Lee 1: `chao_lee1`
- Chao & Lee 2: `chao_lee2`
- Chao Presence/Absence: `chao_pa`
- Chao Quantitative: `chao_quant`
- Convex hull volume: `hull_volume`
- Dalton index: `dalton`
- Equal shares coefficient: `equal_shares_coef`
- Fisher's $\\alpha$: `fisher_alpha`
- Functional Attribute Diversity: `fad`
- Functional Hill Numbers: `func_hill`
- Functional Range: `func_range`
- Functional dendrogram: `func_dendrogram`
- Functional divergence: `func_divergence`
- Functional dispersion: `func_dispersion`
- Functional evenness: `func_evenness`
- Functional logarithmic variance: `func_logvar`
- Functional unalikeability: `func_unalikeability`
- Functional richness: `func_richness`
- Functional variance: `func_var`
- Functional volume: `func_vol`
- Gini inequality index: `gini_coef`
- Help index: `help_index`
- Huffman code length: `huffman`
- Leinster-Cobbold index: `leinster_cobbold`
- Lempel-Ziv code length: `lempel_ziv`
- Logarithmic variance: `logvar`
- Lorenz curves: `lorenz_curve`
- Magurran-Henderson: `magurran_henderson`
- Margalef's index: `margalef`
- McIntosh's D: `mcintosh_d`
- McIntosh's E: `mcintosh_e`
- Michaelis-Menten: `michaelis_menten`
- Minimal majority: `min_majority`
- (MNDif) Mean absolute pairwise difference: `MNDif`
- Menhinick's index: `menhinick`
- Modified functional attribute diversity: `mfad`
- Pielou J: `pielou`
- Pietra index: `pietra`
- Pooled Rarefaction: `pooled_rarefaction`
- Relative mean deviation: `rel_mean_dev`
- Rao's (generalized) quadratic entropy: `rqe`
- Numbers equivalent RQE: `rqe_neq`
- Rademacher complexity: `rademacher`
- Ricotta-Szeidl entropy: `ricotta_szeidl`
- Simpson's E: `simpson_evenness`
- Rarefaction (across samples): `rarefaction`
- Rarefaction (single sample): `rarefaction_single`
- Smith & Wilson B: `smith_wilson_b`
- Smith & Wilson 1D: `smith_wilson_1d`
- Smith & Wilson -lnD: `smith_wilson_lnd`
- StDev1 & StDev2: `StDev1`, `StDev2`
- Strong's index: `strong_index`
- Variance of logarithms: `var_logs`
- VarNC: `VarNC`

- We will also add wrappers for existing functions to account for synonymous indices such as
    - Hartley heterogeneity and "species richness": `species_richness`
    - and more
