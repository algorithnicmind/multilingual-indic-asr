"""
Setup script for Multilingual Indic ASR.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="multilingual-indic-asr",
    version="1.0.0",
    author="Ankit",
    description="A from-scratch multilingual ASR system for English, Hindi, and Odia",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/multilingual-indic-asr",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.1",
        "scikit-learn>=1.2.0",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "flake8>=6.0.0",
            "black>=23.0.0",
        ],
        "ui": [
            "sounddevice>=0.4.6",
        ],
    },
    entry_points={
        "console_scripts": [
            "indic-asr=src.inference:transcribe",
            "indic-asr-ui=ui.app:run_app",
        ],
    },
)
