from setuptools import setup

with open("README.md") as file:
    long_desc = file.read()

setup(
    name='spectral_analysis',
    version='0.0.1',
    description='Library and scripts for processing, analysis, and visualization of hyperspectral images.',
    long_description = long_desc,
    long_description_content_type='text/markdown',
    py_modules=["spectral_io"],
    package_dir={
        '': 'src'
    }
)

# TODO, add licence, add classifiers, go through setup attribute docs, set up docs, setup high level module like "spyctral", add min requirements, add url and author details, do the pip install manifest thing, setup tox.ini and manifest.ini, setup travis.yml or .gitlab-ci.yml
# TODO, extra credit, add code coverage, quality metrics, version management, multiplatform testing, expand docs, consider setup.cfg or pyproject.toml
