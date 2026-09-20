"""Sphinx configuration for the pyFragility documentation."""

import importlib.metadata

project = "pyFragility"
author = "Laxman Dahal"
copyright = "2024-2026, Laxman Dahal"
release = importlib.metadata.version("pyFragility")
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints", "Thumbs.db", ".DS_Store"]
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}
master_doc = "index"

# --- API documentation ---------------------------------------------------------------------
autosummary_generate = True
autodoc_typehints = "none"  # parameter types are documented in the numpydoc sections
autodoc_member_order = "bysource"
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = False
numpydoc_validation_checks = {"PR01", "PR02"}
# Result containers are documented through "Attributes" (users never construct them).
numpydoc_validation_exclude = {
    r"\.__",
    r"\.(FragilityFit|LognormalSummary|CovarianceEstimates|ProbitGLMResult|MLEResult|TestResult"
    r"|GoodnessOfFit|BootstrapResult|PosteriorSamples|FrequencyUncertainty"
    r"|CollapseRateSimulation|MSADataset)$",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "statsmodels": ("https://www.statsmodels.org/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# --- Markdown / notebooks -----------------------------------------------------------------
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_heading_anchors = 3
nb_execution_mode = "auto"  # run code cells of Markdown pages; keep stored notebook outputs
nb_execution_timeout = 300
nb_execution_raise_on_error = True

# --- HTML ---------------------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = f"pyFragility {version}"
html_static_path = ["_static"]
html_theme_options = {
    "github_url": "https://github.com/laxmandahal/pyFragility",
    "navigation_with_keys": False,
    "show_toc_level": 2,
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/pyFragility/",
            "icon": "fa-brands fa-python",
        }
    ],
}
html_context = {"default_mode": "light"}
