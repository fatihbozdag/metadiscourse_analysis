"""
Setup configuration for the Metalinguistics package
"""

from setuptools import setup, find_packages
import os

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="metalinguistics",
    version="0.1.0",
    author="Fatih Bozdag",
    author_email="your.email@example.com",
    description="Advanced Metadiscourse Analysis Library using NLP and Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/metalinguistics",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "metalinguistics=metalinguistics.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "metalinguistics": ["config/**/*.json", "config/**/*.yaml"],
    },
    zip_safe=False,
)