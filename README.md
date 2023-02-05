# ⚠️ This repository is no longer actively developed. <br />  Future development will continue in [this fork](https://github.com/jbrinchmann/metaldisc).


# Forward modelling of galaxy metallicity profiles

This code provides a method for recovering gas-phase metallicity gradients from integral field spectroscopic (IFS) observations of barely resolved galaxies.

I employ a forward modelling approach to fit the observed spatial distribution of emission-line fluxes, accounting for the degrading effects of seeing and spatial binning.
The method is flexible and is not limited to particular emission lines or instruments, nor one set specific set of photoionization models.
outline

While this model can be fit in many ways, we adopt a Bayesian approach with a robust likelihood.

In the `example` directory, there are two scripts:
 - `fit_from_sfrmap.py` - shows how to fit the example data
 - `model_from_sfrmap.py` - provides a visualisation of the model output

## Citing
If you use this code, you should cite [Carton et al. 2017](https://ui.adsabs.harvard.edu/abs/2017MNRAS.468.2140C), where the method is explained.

You may also wish to cite [Carton et al. 2018](https://ui.adsabs.harvard.edu/abs/2018MNRAS.478.4293C), where we provide further discussion and improvements.


## GNU GPL v3
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
