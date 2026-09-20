"""Fit a collapse fragility to MSA data and compare MLE and sandwich (QMLE) uncertainty."""

import numpy as np

from pyFragility import (
    CollapseData,
    collapse_frequency_std,
    covariance_estimates,
    fit_mle,
    fit_probit_glm,
    mean_annual_collapse_frequency,
    probability_of_collapse_in_years,
)


def main() -> None:
    im = [0.178, 0.274, 0.444, 0.56, 0.652, 0.79, 0.982, 1.246]
    im += [1.564, 2.014, 2.417, 3.021, 3.625, 4.028, 4.431, 5.035]
    return_periods = [15, 25, 50, 75, 100, 150, 250, 500, 1000, 2500, 2700, 3000]
    return_periods += [3300, 3500, 3700, 4000]
    counts = [0, 0, 0, 0, 0, 4, 13, 23, 38, 41, 44, 45, 45, 45, 45, 45]
    data = CollapseData.from_return_periods(im, counts, 45 * np.ones(len(im)), return_periods)

    fragility = fit_mle(data).fragility
    print(f"theta (median) = {fragility.theta:.4f}, beta (log-std) = {fragility.beta:.4f}")

    cov = covariance_estimates(data, fragility)
    print("Var[theta, beta] (MLE):     ", np.diag(cov.mle_cov))
    print("Var[theta, beta] (sandwich):", np.diag(cov.sandwich_cov))

    # Collapse risk: uncertainty from the GLM covariance, with and without the sandwich.
    for label, cov_type in [("MLE", "nonrobust"), ("QMLE", "expected_hessian")]:
        glm = fit_probit_glm(data, cov_type)
        std = collapse_frequency_std(glm.fragility, glm.cov, data)
        print(f"MAFC std ({label}): {std:.3e}")

    mafc = mean_annual_collapse_frequency(fragility.probability, data)
    print(
        f"MAFC = {mafc:.3e}, 50-yr collapse probability = "
        f"{probability_of_collapse_in_years(mafc, 50):.4f}"
    )


if __name__ == "__main__":
    main()
