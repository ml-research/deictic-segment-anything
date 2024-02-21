import os

from setuptools import find_packages, setup

# import ipdb; ipdb.set_trace()


def package_files(directory):
    paths = []
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
    return paths


# extra_files = package_files('nsfr/data') + package_files('nsfr/lark')
extra_files = package_files("neumann/src/lark")
# import ipdb; ipdb.set_trace()

setup(
    name="neumann",
    version="0.1.0",
    author="anonymous",
    author_email="anon.gouv",
    packages=find_packages(),
    package_data={"": extra_files},
    include_package_data=True,
    # package_dir={'':'src'},
    url="tba",
    description="GNN-based Differentiable Forward Reasoner",
    long_description=open("README.md").read(),
    install_requires=[
        "matplotlib",
        "numpy",
        "seaborn",
        "setuptools",
        #        "torch",
        "tqdm",
        "lark",
    ],
)
