from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pathsim",
    version="0.4.2",
    author="Milan Rother",
    author_email="milan.rother@gmx.de",
    description="A block based time domain system simulation framework.",
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