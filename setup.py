#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for CERT SDK.

CERT: Instrumentation and Metrics for Production LLM Sequential Processing

This setup.py provides compatibility for tools that don't support pyproject.toml.
For modern installations, pyproject.toml is the preferred configuration.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Core dependencies
INSTALL_REQUIRES = [
    "numpy>=1.24.0",
    "scipy>=1.11.0",
    "sentence-transformers>=2.2.0",
    "anthropic>=0.7.0",
    "openai>=1.0.0",
    "google-generativeai>=0.3.0",
    "streamlit>=1.28.0",
    "prometheus-client>=0.19.0",
    "pydantic>=2.0.0",
    "aiohttp>=3.9.0",
    "tenacity>=8.2.0",
    "pandas>=2.0.0",
    "plotly>=5.17.0",
]

# Development dependencies
DEV_REQUIRES = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.5.0",
    "types-requests>=2.31.0",
]

setup(
    name="cert-sdk",
    version="0.1.0",
    description="Observability infrastructure for multi-model LLM sequential processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Javier Marín",
    author_email="javier@jmarin.info",
    url="https://github.com/Javihaus/CERT",
    project_urls={
        "Homepage": "https://github.com/Javihaus/CERT",
        "Documentation": "https://cert-sdk.readthedocs.io",
        "Repository": "https://github.com/Javihaus/CERT",
        "Issues": "https://github.com/Javihaus/CERT/issues",
    },
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "dev": DEV_REQUIRES,
    },
    keywords=[
        "observability",
        "multi-model",
        "llm",
        "production",
        "sequential-processing",
        "attention-mechanisms",
        "mlsys",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False,
    include_package_data=True,
)
