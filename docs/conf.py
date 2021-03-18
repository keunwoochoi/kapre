
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../'))
import sphinx_rtd_theme

autodoc_mock_imports = ['tensorflow', 'librosa', 'numpy']
autodoc_member_order = 'bysource'


# -- Project information -----------------------------------------------------

project = 'Kapre'
copyright = '2020, Keunwoo Choi, Deokjin Joo and Juho Kim'
author = 'Keunwoo Choi, Deokjin Joo and Juho Kim'

# The full version, including alpha/beta/rc tags
release = '2017'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx_rtd_theme",
    "sphinx.ext.viewcode",  # source linkage
    "sphinx.ext.napoleon",
    # "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",  # source linkage
    "sphinxcontrib.inlinesyntaxhighlight"  # inline code highlight
]

# https://stackoverflow.com/questions/21591107/sphinx-inline-code-highlight
# use language set by highlight directive if no language is set by role
inline_highlight_respect_highlight = True
# use language set by highlight directive if no role is set
inline_highlight_literals = True

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# autosummary_generate = True

# autoapi_type = 'python'
# autoapi_dirs = ['../kapre']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    'css/custom.css',
]


def setup(app):
    app.add_stylesheet("css/custom.css")


master_doc = 'index'

