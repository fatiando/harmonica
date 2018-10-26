# -*- coding: utf-8 -*-
import sys
import os
import datetime
import sphinx_rtd_theme
import sphinx_gallery
from sphinx_gallery.sorting import FileNameSortKey

# Sphinx needs to be able to import the package to use autodoc and get the
# version number
sys.path.append(os.path.pardir)

from harmonica.version import full_version

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.doctest',
    'sphinx.ext.viewcode',
    'sphinx.ext.extlinks',
    "sphinx.ext.intersphinx",
    'matplotlib.sphinxext.plot_directive',
    'numpydoc',
    'sphinx_gallery.gen_gallery',
]

# intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/stable/", None),
    "xarray": ("http://xarray.pydata.org/en/stable/", None),
    "cartopy": ("https://scitools.org.uk/cartopy/docs/latest/", None),
    "pooch": ("https://www.fatiando.org/pooch/latest/", None),
    "verde": ("https://www.fatiando.org/verde/latest/", None),
    "matplotlib": ("https://matplotlib.org/", None),
}

# Autosummary pages will be generated by sphinx-autogen instead of sphinx-build
autosummary_generate = False

numpydoc_class_members_toctree = False

sphinx_gallery_conf = {
    # path to your examples scripts
    'examples_dirs': ['../examples'],
    # path where to save gallery generated examples
    'gallery_dirs': ['gallery'],
    'filename_pattern': '\.py',
    # Remove the "Download all examples" button from the top level gallery
    'download_all_examples': False,
    # Sort gallery example by file name instead of number of lines (default)
    'within_subsection_order': FileNameSortKey,
    # directory where function granular galleries are stored
    'backreferences_dir': 'api/generated/backreferences',
    # Modules for which function level galleries are created.  In
    # this case sphinx_gallery and numpy in a tuple of strings.
    'doc_module': 'harmonica',
    # Insert links to documentation of objects in the examples
    'reference_url': {'harmonica': None},
}

# Always show the source code that generates a plot
plot_include_source = True
plot_formats = ['png']

# Sphinx project configuration
templates_path = ['_templates']
exclude_patterns = ['_build', '**.ipynb_checkpoints']
source_suffix = '.rst'
# The encoding of source files.
source_encoding = 'utf-8-sig'
master_doc = 'index'

# General information about the project
year = datetime.date.today().year
project = 'Harmonica'
copyright = '2018-{}, Leonardo Uieda'.format(year)
if len(full_version.split('+')) > 1 or full_version == 'unknown':
    version = 'dev'
else:
    version = full_version

# These enable substitutions using |variable| in the rst files
rst_epilog = """
.. |year| replace:: {year}
""".format(year=year)

html_last_updated_fmt = '%b %d, %Y'
html_title = project
html_short_title = project
html_logo = '_static/harmonica-logo.png'
html_favicon = '_static/favicon.png'
html_static_path = ['_static']
html_extra_path = []
pygments_style = 'default'
add_function_parentheses = False
html_show_sourcelink = False
html_show_sphinx = True
html_show_copyright = True

# Theme config
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    'logo_only': True,
    'display_version': True,
}
html_context = {
    'menu_links_name': 'Getting help and contributing',
    'menu_links': [
        ('<i class="fa fa-external-link-square fa-fw"></i> Fatiando a Terra', 'https://www.fatiando.org'),
        ('<i class="fa fa-users fa-fw"></i> Contributing', 'https://github.com/fatiando/harmonica/blob/master/CONTRIBUTING.md'),
        ('<i class="fa fa-gavel fa-fw"></i> Code of Conduct', 'https://github.com/fatiando/harmonica/blob/master/CODE_OF_CONDUCT.md'),
        ('<i class="fa fa-comment fa-fw"></i> Contact', 'https://gitter.im/fatiando/fatiando'),
        ('<i class="fa fa-github fa-fw"></i> Source Code', 'https://github.com/fatiando/harmonica'),
    ],
    # Custom variables to enable "Improve this page"" and "Download notebook"
    # links
    'doc_path': 'doc',
    'galleries': sphinx_gallery_conf['gallery_dirs'],
    'gallery_dir': dict(zip(sphinx_gallery_conf['gallery_dirs'],
                            sphinx_gallery_conf['examples_dirs'])),
    'github_repo': 'fatiando/harmonica',
    'github_version': 'master',
}

# Load the custom CSS files (needs sphinx >= 1.6 for this to work)
def setup(app):
    app.add_stylesheet("style.css")
