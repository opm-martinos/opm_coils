# # Minimal makefile for Sphinx documentation
# #
#
# # You can set these variables from the command line, and also
# # from the environment for the first two.
# SPHINXOPTS    ?=
# SPHINXBUILD   ?= sphinx-build
# SOURCEDIR     = .
# BUILDDIR      = _build
#
# # Put it first so that "make" without argument is like "make help".
# help:
# 	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
#
# .PHONY: help Makefile
#
# # Catch-all target: route all unknown targets to Sphinx using the new
# # "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
# %: Makefile
# 	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
#
#
# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    = -v
SPHINXBUILD   = sphinx-build
PAPER         =
BUILDDIR      = _build

# User-friendly check for sphinx-build
ifeq ($(shell which $(SPHINXBUILD) >/dev/null 2>&1; echo $$?), 1)
$(error The '$(SPHINXBUILD)' command was not found. Make sure you have Sphinx installed, then set the SPHINXBUILD environment variable to point to the full path of the '$(SPHINXBUILD)' executable. Alternatively you can add the directory with the executable to your PATH. If you don't have Sphinx installed, grab it from http://sphinx-doc.org/)
endif

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .
# the i18n builder cannot share the environment and doctrees with the others
I18NSPHINXOPTS  = $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html-noplot to make standalone HTML files, without plotting anything"
	@echo "  html       to make standalone HTML files"
	@echo "  html_dev-pattern to make standalone HTML files for one example dir (dev version)"
	@ech "   view	    to view the documentation in a browser"
	@echo "  linkcheck        to check all external links for integrity"

.PHONY: clean

clean:
	rm -rf $(BUILDDIR)/*
	rm -rf auto_examples/
	rm -rf generated/*
	rm -rf modules/*

html-noplot:
	$(SPHINXBUILD) -D plot_gallery=0 -D nbsphinx_execute=never -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

.PHONY: html
html:
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

html_dev-pattern:
	BUILD_DEV_HTML=1 $(SPHINXBUILD) -D sphinx_gallery_conf.filename_pattern=$(PATTERN) -D sphinx_gallery_conf.run_stale_examples=True -b html $(ALLSPHINXOPTS) _build/html
	@echo
	@echo "Build finished. The HTML pages are in _build/html"

linkcheck:
	@$(SPHINXBUILD) -b linkcheck -D nitpicky=0 -D plot_gallery=0 -d _build/doctrees . _build/linkcheck

view:
	@python -c "import webbrowser; webbrowser.open_new_tab('file://$(PWD)/_build/html/index.html')"

install:
	rm -rf _build/doctrees _build/nulling_coils
	git clone -b main --single-branch --no-checkout --depth 1 git@github.com:opm-martinos/nulling_coils.git _build/nulling_coils
	touch _build/nulling_coils/.nojekyll
	make html
	cd _build/ && \
	cp -r html/* nulling_coils && \
	cd nulling_coils && \
	git add * && \
	git add .nojekyll && \
	git commit -a -m 'Update webpage' && \
	git push
