# Installation script for python
import os
import re

from setuptools import find_packages, setup

PACKAGE = "qrccoherence"


# Returns the qibo version
def get_version():
    """Gets the version from the package's __init__ file
    if there is some problem, let it happily fail"""
    VERSIONFILE = os.path.join("src", PACKAGE, "__init__.py")
    initfile_lines = open(VERSIONFILE, "rt").readlines()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)


# Read in requirements
requirements = open("requirements.txt").readlines()
requirements = [r.strip() for r in requirements]


# load long description from README
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="QRCcoherence",
    version=get_version(),
    description="Project studing the importance of coherence in Quantum Reservoir Computing (QRC)",
    author="Ana Palacios",
    author_email="anapaluis@hotmail.com",
    url="https://github.com/anapalu/QRCcoherence",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"": ["*.out"]},
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    install_requires=requirements,
    extras_require={
        "docs": [
            "sphinx",
            "sphinx_rtd_theme",
            "recommonmark",
            "sphinxcontrib-bibtex",
            "sphinx_markdown_tables",
            "nbsphinx",
        ],
        "tests": ["pytest"],
    },
    python_requires=">=3.8",
    long_description=long_description,
    long_description_content_type="text/markdown",
)

