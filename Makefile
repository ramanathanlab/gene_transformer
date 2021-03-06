.DEFAULT_GOAL := all
isort = isort gene_transformer
black = black --target-version py37 gene_transformer

.PHONY: format
format:
	$(isort)
	$(black)

.PHONY: lint
lint:
	$(black) --check --diff
	flake8 gene_transformer/

.PHONY: mypy
mypy:
	mypy --config-file setup.cfg --package gene_transformer
	mypy --config-file setup.cfg examples/

.PHONY: all
all: format lint mypy