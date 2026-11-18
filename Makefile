# Build, package, test, and clean
STYLE_CHECK_FILES=src test doc
GITHUB_ACTIONS=.github/workflows

.PHONY: build help install test format check check-format check-style check-actions clean

help:
	@echo "Commands:"
	@echo ""
	@echo "  install   install in editable mode"
	@echo "  test      run the test suite (including doctests) and report coverage"
	@echo "  format    run ruff to automatically format the code"
	@echo "  check     run code style and quality checks (ruff and burocrata)"
	@echo "  build     build source and wheel distributions"
	@echo "  clean     clean up build and generated files"
	@echo ""

build:
	python -m build .

install:
	python -m pip install --no-deps -e .

test:
	NUMBA_JIT_COVERAGE=1 pytest

format:
	ruff check --select I --fix $(STYLE_CHECK_FILES) # fix isort errors
	ruff format $(STYLE_CHECK_FILES)
	burocrata --extension=py $(STYLE_CHECK_FILES)

check: check-format check-style check-actions

check-format:
	ruff format --check $(STYLE_CHECK_FILES)
	burocrata --check --extension=py $(STYLE_CHECK_FILES)

check-style:
	ruff check $(STYLE_CHECK_FILES)

check-actions:
	zizmor $(GITHUB_ACTIONS)

clean:
	find . -name "*.pyc" -exec rm -v "{}" \;
	find . -name "*.orig" -exec rm -v "{}" \;
	find . -name ".coverage.*" -exec rm -v "{}" \;
	find . -name "_version.py" -exec rm -v "{}" \;
	find . -name "*.egg-info" -type d -exec rm -vr "{}" \; -prune
	find . -name "__pycache__" -type d -exec rm -vr "{}" \; -prune
	rm -rvf build dist MANIFEST .coverage .cache .pytest_cache
