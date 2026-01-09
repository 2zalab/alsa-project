"""
Setup script for A-LSA package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="alsa-classifier",
    version="1.0.0",
    author="Isaac Touza",
    author_email="isaac.touza@univ-maroua.cm",
    description="Adaptive Latent Semantic Analysis for Binary Text Classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/alsa-project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
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
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "nltk>=3.6",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "psutil>=5.8.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "ipywidgets>=7.6.0",
            "pytest>=6.0.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
        "bert": [
            "transformers>=4.0.0",
            "torch>=1.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "alsa-sms=experiments.run_sms_spam:main",
            "alsa-imdb=experiments.run_imdb:main",
            "alsa-newsgroups=experiments.run_newsgroups:main",
            "alsa-sensitivity=experiments.sensitivity_analysis:main",
        ],
    },
)
