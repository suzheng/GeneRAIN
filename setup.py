from setuptools import setup, find_packages

setup(
    name="GeneRAIN",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
