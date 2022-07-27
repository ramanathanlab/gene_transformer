.DEFAULT_GOAL := all
package_name = gene_transformer
extra_folders = #examples/ test/
isort = isort $(package_name) $(extra_folders)
black = black --target-version py37 $(package_name) $(extra_folders)
flake8 = flake8 $(package_name)/ $(extra_folders)
pylint = pylint $(package_name)/ $(extra_folders)
pydocstyle = pydocstyle $(package_name)/
run_mypy = mypy --config-file setup.cfg


.PHONY: format
format:
	$(isort)
	$(black)

.PHONY: lint
lint:
	$(black) --check --diff
	$(flake8)
	#$(pylint)
	#$(pydocstyle)

.PHONY: mypy
mypy:
	$(run_mypy) --package $(package_name)
	$(run_mypy) $(package_name)/
	$(run_mypy) $(extra_folders)

.PHONY: all
all: format lint #mypy