from importlib import metadata

project = "Bergson"
copyright = "2026, Anonymous Authors"
html_title = "Bergson"
html_logo = "_static/bergson_logo.png"
html_favicon = "_static/favicon.ico"

author = "Anonymous Authors"

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx_autodoc_typehints",
    "sphinx.ext.doctest",
    "myst_parser",
    "nbsphinx",
]

release = metadata.version("Bergson")

# Prefix auto-generated section labels with the document name so same-named
# headings in different pages (e.g. "Quickstart") don't collide.
autosectionlabel_prefix_document = True

napoleon_google_docstring = True
napoleon_use_param = False
napoleon_use_ivar = True

templates_path = ["_templates"]
exclude_patterns = ["build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

html_theme = "furo"
html_static_path = ["_static"]
html_theme_options = {
    "light_css_variables": {
        "sidebar-item-font-size": "85%",
    },
}
