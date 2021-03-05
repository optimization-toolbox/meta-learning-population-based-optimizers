import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="meta-learning-optimizers",
    version="0.1",
    author="Hugo Dovs",
    author_email="hugodovs@gmail.com",
    description="MLO: Meta Learning Optimizers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Hugodovs/meta-learning-optimizers",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GNU General Public License v3.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    install_requires=["pyyaml", "numpy", "matplotlib", "cma", "torch", "mpi4py"]
)