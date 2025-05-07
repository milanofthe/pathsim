import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))


from pathsim import __version__ 

# -- Project information -----------------------------------------------------

project = 'pathsim'
copyright = '2025, Milan Rother' 
author = 'Milan Rother'
version = __version__
release = __version__

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',  # Core Sphinx library for auto doc generation
    'sphinx.ext.napoleon', # Support for NumPy and Google style docstrings
    'sphinx.ext.viewcode',  # Add links to source code
    'sphinx.ext.mathjax', # Render math
    'myst_parser',          # Support for MyST Markdown (optional, but recommended)
	'sphinx.ext.autosummary', # Create neat summary tables,
    'sphinx.ext.intersphinx', 
    'sphinx_rtd_theme',
    'sphinx_copybutton',
]

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'  
html_static_path = ['_static']
html_logo = 'logos/pathsim_logo_mono_lg.png'
html_theme_options = {
    'logo_only' : True,
    'display_version' : False,
}

# -- Options for autodoc -----------------------------------------------------

autodoc_default_options = {
    'members': True,       # Document all members (functions, classes, methods)
    'member-order': 'bysource', # Order members as they appear in the source code
    'undoc-members': True,  # Include members that don't have docstrings
    'show-inheritance': True, # Show base classes
}
autosummary_generate = True  # Turn on sphinx.ext.autosummary

# -- Options for MyST Parser -----------------------------------------------

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

myst_update_mathjax = False
myst_heading_anchors = 3  
myst_url_schemes = ("http", "https") 
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "linkify",
]

# Add support to link variables in other projects, used in the docstrings
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}