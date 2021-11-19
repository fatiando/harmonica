# Build, package, test, and clean
PROJECT=harmonica
TESTDIR=tmp-test-dir-with-unique-name
PYTEST_ARGS=--cov-config=../.coveragerc --cov-report=term-missing --cov=$(PROJECT) --doctest-modules -v --pyargs
NUMBATEST_ARGS=--doctest-modules -v --pyargs -m use_numba
LINT_FILES=setup.py $(PROJECT) license_notice.py
STYLE_CHECK_FILES=setup.py $(PROJECT) examples data/examples doc/conf.py license_notice.py

help:
	@echo "Commands:"
	@echo ""
	@echo "  install   install in editable mode"
	@echo "  test      run the test suite (including doctests) and report coverage"
	@echo "  format    run isort and black to automatically format the code"
	@echo "  check     run code style and quality checks (black, isort and flake8)"
	@echo "  clean     clean up build and generated files"
	@echo ""

install:
	pip install --no-deps -e .

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

format: license isort black

check: isort-check black-check license-check flake8

black:
	black $(STYLE_CHECK_FILES)

black-check:
	black --check $(STYLE_CHECK_FILES)

isort:
	isort $(STYLE_CHECK_FILES)

isort-check:
	isort --check $(STYLE_CHECK_FILES)

license:
	python license_notice.py

license-check:
	python license_notice.py --check

flake8:
	flake8 $(STYLE_CHECK_FILES)

clean:
	find . -name "*.pyc" -exec rm -v {} \;
	find . -name ".coverage.*" -exec rm -v {} \;
	rm -rvf build dist MANIFEST *.egg-info __pycache__ .coverage .cache .pytest_cache $(PROJECT)/_version.py
	rm -rvf $(TESTDIR) dask-worker-space
