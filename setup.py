from setuptools import setup, find_packages
import os
import re

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def get_version():
    version_file = os.path.join("pathsim", "_version.py")
    with open(version_file, "r") as f:
        version_line = f.read()    
    version_regex = r"__version__ = ['\"]([^'\"]*)['\"]"
    match = re.search(version_regex, version_line)
    if match:
        return match.group(1)
    raise RuntimeError(f"Unable to find version string in {version_file}")


setup(
    name="pathsim",
    version=get_version(),
    author="Milan Rother",
    author_email="milan.rother@gmx.de",
    description="A differentiable block based hybrid system simulation framework.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/milanofthe/pathsim",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
    ],
)