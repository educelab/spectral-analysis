from setuptools import setup

with open("README.md") as file:
    long_desc = file.read()

setup(
    name='spectral_analysis',
    version='0.0.1',
    description='Library and scripts for processing, analysis, and visualization of spectral images.',
    long_description=long_desc,
    long_description_content_type='text/markdown',
    packages=["spectral_analysis"],
    install_requires=['imageio', 'numpy'],
    entry_points={"console_scripts": [
        "spectral_manager = spectral_analysis.scripts.spectral_manager:main"
    ]}
)