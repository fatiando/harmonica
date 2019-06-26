# Build, package, test, and clean
PROJECT=harmonica
TESTDIR=tmp-test-dir-with-unique-name
PYTEST_ARGS=--cov-config=../.coveragerc --cov-report=term-missing --cov=$(PROJECT) --doctest-modules -v --pyargs
NUMBATEST_ARGS=--doctest-modules -v --pyargs -m use_numba
LINT_FILES=setup.py $(PROJECT)
BLACK_FILES=setup.py $(PROJECT) examples data/examples
FLAKE8_FILES=setup.py $(PROJECT) examples data/examples

help:
	@echo "Commands:"
	@echo ""
	@echo "  install   install in editable mode"
	@echo "  test      run the test suite (including doctests) and report coverage"
	@echo "  format    run black to automatically format the code"
	@echo "  check     run code style and quality checks (black and flake8)"
	@echo "  lint      run pylint for a deeper (and slower) quality check"
	@echo "  clean     clean up build and generated files"
	@echo ""

install:
	pip install --no-deps -e .

test: test_coverage test_numba

test_coverage:
	# Run a tmp folder to make sure the tests are run on the installed version
	mkdir -p $(TESTDIR)
	cd $(TESTDIR); export NUMBA_DISABLE_JIT=1; MPLBACKEND='agg' pytest $(PYTEST_ARGS) $(PROJECT)
	cp $(TESTDIR)/.coverage* .
	rm -rvf $(TESTDIR)

test_numba:
	# Run a tmp folder to make sure the tests are run on the installed version
	export NUMBA_DISABLE_JIT=0; MPLBACKEND='agg' pytest $(NUMBATEST_ARGS) $(PROJECT)

format:
	black $(BLACK_FILES)

check:
	black --check $(BLACK_FILES)
	flake8 $(FLAKE8_FILES)

lint:
	pylint --jobs=0 $(LINT_FILES)

clean:
	find . -name "*.pyc" -exec rm -v {} \;
	find . -name ".coverage.*" -exec rm -v {} \;
	rm -rvf build dist MANIFEST *.egg-info __pycache__ .coverage .cache .pytest_cache
	rm -rvf $(TESTDIR) dask-worker-space
