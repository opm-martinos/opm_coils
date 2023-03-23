# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from sphinx_gallery.sorting import ExampleTitleSortKey 
import numpy as np
import pyvista


project = 'opmcoils'
copyright = '2023, Mainak Jas'
author = 'Mainak Jas'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_gallery.gen_gallery',
    # 'sphinx.ext.viewcode',  # uncomment when we open source the code
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'numpydoc',
    'sphinx_copybutton',
]

# necessary when building the sphinx gallery
pyvista.BUILDING_GALLERY = True
pyvista.OFF_SCREEN = True

# Optional - set parameters like theme or window size
pyvista.set_plot_theme('document')
pyvista.global_theme.window_size = np.array([1024, 768]) * 2

autosummary_generate = True
autodoc_default_options = {'inherited-members': None}
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = True
default_role = 'autolink'

# Sphinx-Copybutton configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store',
                    "auto_examples/*.ipynb",
                    "auto_examples/*.py"]

source_suffix = '.rst'

pygments_style = 'sphinx'

sphinx_gallery_conf = {
    "doc_module": ("opmcoils"),
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    'backreferences_dir': 'generated',
    'ignore_pattern': '^(?!.*plot_).*$',
    "image_scrapers": ('matplotlib', 'pyvista'),
    'within_subsection_order': ExampleTitleSortKey
}


intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'mne': ('https://mne.tools/dev', None),
    'numpy': ('https://numpy.org/devdocs', None),
    'scipy': ('https://scipy.github.io/devdocs', None),
    'matplotlib': ('https://matplotlib.org', None)
}
intersphinx_timeout = 5

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']

html_theme_options = {
    "logo": {
        "text": "opmcoils",
    },
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/opm-martinos/cmeg_coil_design",
            "icon": "fab fa-github-square",
        },
    ],
    "secondary_sidebar_items": ["page-toc"],
    "show_toc_level": 1,
    "show_prev_next": False,
}

html_sidebars = {
  "assembly": []
}
