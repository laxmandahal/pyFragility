"""Link (inverse-CDF) functions for binomial and ordinal fragility models.

Each link maps the linear predictor ``eta`` to a probability ``F(eta)`` and exposes the
log-probabilities and density derivatives needed for analytic likelihood derivatives.
"""

from __future__ import annotations

import numpy as np
from scipy import special
from scipy.stats import norm


class Link:
    name: str

    def cdf(self, eta):
        raise NotImplementedError

    def log_cdf(self, eta):
        raise NotImplementedError

    def log_sf(self, eta):
        raise NotImplementedError

    def log_pdf(self, eta):
        raise NotImplementedError

    def dlog_pdf(self, eta):
        """``f'(eta) / f(eta)``."""
        raise NotImplementedError

    def ppf(self, p):
        """Inverse of :meth:`cdf`."""
        raise NotImplementedError


class Probit(Link):
    name = "probit"

    def cdf(self, eta):
        return special.ndtr(eta)

    def log_cdf(self, eta):
        return special.log_ndtr(eta)

    def log_sf(self, eta):
        return special.log_ndtr(-np.asarray(eta))

    def log_pdf(self, eta):
        return norm.logpdf(eta)

    def dlog_pdf(self, eta):
        return -np.asarray(eta)

    def ppf(self, p):
        return special.ndtri(p)


class Logit(Link):
    name = "logit"

    def cdf(self, eta):
        return special.expit(eta)

    def log_cdf(self, eta):
        return special.log_expit(eta)

    def log_sf(self, eta):
        return special.log_expit(-np.asarray(eta))

    def log_pdf(self, eta):
        return special.log_expit(eta) + special.log_expit(-np.asarray(eta))

    def dlog_pdf(self, eta):
        return 1.0 - 2.0 * special.expit(eta)

    def ppf(self, p):
        return special.logit(p)


class Cloglog(Link):
    """Complementary log-log: ``F(eta) = 1 - exp(-exp(eta))`` (asymmetric, Gumbel-type)."""

    name = "cloglog"

    @staticmethod
    def _t(eta):
        return np.exp(np.clip(eta, -700.0, 30.0))

    def cdf(self, eta):
        return -np.expm1(-self._t(eta))

    def log_cdf(self, eta):
        return np.log(-np.expm1(-self._t(eta)))

    def log_sf(self, eta):
        return -self._t(eta)

    def log_pdf(self, eta):
        return np.clip(eta, -700.0, 30.0) - self._t(eta)

    def dlog_pdf(self, eta):
        return 1.0 - self._t(eta)

    def ppf(self, p):
        return np.log(-np.log1p(-np.asarray(p)))


LINKS: dict[str, Link] = {"probit": Probit(), "logit": Logit(), "cloglog": Cloglog()}


def get_link(link: str | Link) -> Link:
    """Return a :class:`Link` from its name (probit, logit, cloglog) or pass one through."""
    if isinstance(link, Link):
        return link
    try:
        return LINKS[link]
    except KeyError:
        raise ValueError(f"unknown link {link!r}; choose from {sorted(LINKS)}") from None
