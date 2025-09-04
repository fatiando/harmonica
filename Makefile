# Build, package, test, and clean
PROJECT=harmonica
TESTDIR=tmp-test-dir-with-unique-name
PYTEST_ARGS=--cov-config=../.coveragerc --cov-report=term-missing --cov=$(PROJECT) --doctest-modules --doctest-continue-on-failure -v --pyargs
NUMBATEST_ARGS=--doctest-modules -v --pyargs -m use_numba
STYLE_CHECK_FILES=$(PROJECT) doc
GITHUB_ACTIONS=.github/workflows

.PHONY: build install test test_coverage test_numba format check check-format check-style check-actions clean

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

.PHONY: build, install, test, test_coverage, test_numba, format, check, check-format, check-style, check-actions, clean

build:
	python -m build .

install:
	python -m pip install --no-deps -e .

test: test_coverage test_numba

test_coverage:
	# Run a tmp folder to make sure the tests are run on the installed version
	mkdir -p $(TESTDIR)
	cd $(TESTDIR); NUMBA_DISABLE_JIT=1 MPLBACKEND='agg' pytest $(PYTEST_ARGS) $(PROJECT)
	cp $(TESTDIR)/.coverage* .
	rm -rvf $(TESTDIR)

test_numba:
	# Run a tmp folder to make sure the tests are run on the installed version
	mkdir -p $(TESTDIR)
	cd $(TESTDIR); NUMBA_DISABLE_JIT=0 MPLBACKEND='agg' pytest $(NUMBATEST_ARGS) $(PROJECT)
	rm -rvf $(TESTDIR)

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
	find . -name "*.pyc" -exec rm -v {} \;
	find . -name ".coverage.*" -exec rm -v {} \;
	rm -rvf build dist MANIFEST *.egg-info __pycache__ .coverage .cache .pytest_cache $(PROJECT)/_version.py
	rm -rvf $(TESTDIR) dask-worker-space
