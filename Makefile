clean:
	rm -rf .pytest_cache build dist miniml.egg-info

install:
	python3 setup.py sdist bdist_wheel && \
	python3 -m pip install dist/miniml-1.0.tar.gz && \
	python3 /Users/oniani/git/miniml/examples/example.py
