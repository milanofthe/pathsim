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
    'sphinx_copybutton',
    'sphinx_design',  # Modern design components (cards, tabs, grids)
    'nbsphinx',  # Jupyter notebook support
]

# -- Options for HTML output -------------------------------------------------

html_theme = 'furo'
html_static_path = ['_static']
html_logo = 'logos/pathsim_logo_mono_g.png'
html_title = "PathSim Documentation"

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2962ff",
        "color-brand-content": "#2962ff",
        "font-stack": "system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif",
        "font-stack--monospace": "SFMono-Regular, Menlo, Consolas, Monaco, Liberation Mono, Lucida Console, monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": "#448aff",
        "color-brand-content": "#448aff",
    },
    "sidebar_hide_name": True,  # Hide project name, show logo only
    "navigation_with_keys": True,  # Allow keyboard navigation
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/milanofthe/pathsim",
    "source_branch": "master",
    "source_directory": "docs/source/",
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

# -- Options for nbsphinx -----------------------------------------------

# Execute notebooks before conversion
nbsphinx_execute = 'auto'  # 'always', 'never', or 'auto' (only if no output)

# Timeout for notebook execution (in seconds)
nbsphinx_timeout = 180

# Allow errors in notebooks (useful during development)
nbsphinx_allow_errors = False

# Kernel to use for notebook execution
nbsphinx_kernel_name = 'python3'

# Custom CSS for notebooks in Furo theme
nbsphinx_prolog = """
.. raw:: html

    <style>
        /* Make notebook outputs fit better with Furo theme */
        .nboutput .output_area pre {
            background-color: var(--color-background-secondary);
            border: 1px solid var(--color-background-border);
            border-radius: 0.25rem;
            padding: 0.5rem;
        }

        /* Transparent figure backgrounds */
        .nboutput img {
            background-color: transparent !important;
        }

        /* Better styling for stderr/logging output */
        .nboutput .stderr {
            background-color: #f5f5f5 !important;
            color: #333333 !important;
            border-left: 3px solid #2962ff;
        }
    </style>
"""

# Exclude certain notebook patterns from execution
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'png'}",
    "--InlineBackend.rc={'figure.dpi': 120}",
]