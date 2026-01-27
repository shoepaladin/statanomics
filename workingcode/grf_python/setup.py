"""
Unified setup.py for GRF package.

Supports multiple installation modes:
  - Basic: Numba-accelerated version (no compilation needed)
  - Full: Cython-optimized version (requires C compiler)

Installation modes:
  # Basic install (Numba only - always works)
  pip install -e .

  # Full install with Cython extensions (requires C++ compiler)
  python setup.py build_ext --inplace
  # OR
  pip install -e . --config-settings="--build-option=--with-cython"
"""

from setuptools import setup, find_packages, Extension
import sys
import os

# Check if Cython is available
try:
    from Cython.Build import cythonize
    import numpy as np
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    cythonize = None


def get_extensions():
    """Build extensions only if explicitly requested or Cython available."""
    # Check if user wants Cython build
    want_cython = (
        '--with-cython' in sys.argv or
        'build_ext' in sys.argv or
        any('--build-option' in arg for arg in sys.argv)
    )
    
    if '--with-cython' in sys.argv:
        sys.argv.remove('--with-cython')
    
    if not want_cython:
        print("\n" + "="*70)
        print("Installing GRF with Numba acceleration (no compilation needed)")
        print("="*70)
        print("\nTo install with Cython optimizations, run:")
        print("  python setup.py build_ext --inplace")
        print("="*70 + "\n")
        return []
    
    if not CYTHON_AVAILABLE:
        print("\n" + "="*70)
        print("WARNING: Cython not available!")
        print("="*70)
        print("\nInstall with: pip install cython")
        print("Falling back to Numba-only installation...")
        print("="*70 + "\n")
        return []
    
    print("\n" + "="*70)
    print("Building Cython extensions for maximum performance")
    print("="*70 + "\n")
    
    extensions = [
        Extension(
            "grf._tree_cython",
            ["grf/_tree_cython.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=['-O3'] if sys.platform != 'win32' else ['/O2'],
        ),
        Extension(
            "grf._weights_cython",
            ["grf/_weights_cython.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=['-O3'] if sys.platform != 'win32' else ['/O2'],
        ),
    ]
    
    return cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
        }
    )


# Read requirements
def read_requirements():
    """Read requirements from requirements.txt if it exists."""
    req_file = 'requirements.txt'
    if os.path.exists(req_file):
        with open(req_file) as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return [
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'scikit-learn>=1.0.0',
        'numba>=0.56.0',
        'joblib>=1.0.0',
    ]


# Read long description
def read_long_description():
    """Read README if it exists."""
    readme_file = 'README.md'
    if os.path.exists(readme_file):
        with open(readme_file, encoding='utf-8') as f:
            return f.read()
    return "Generalized Random Forest for causal inference"


setup(
    name="grf",
    version="0.4.0",
    description="Generalized Random Forest for causal inference with Numba and Cython acceleration",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/grf",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=read_requirements(),
    extras_require={
        'cython': ['cython>=0.29.0'],
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.12.0',
            'black>=21.0',
            'flake8>=3.9.0',
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Statistics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Cython",
    ],
    keywords="causal-inference machine-learning random-forest heterogeneous-treatment-effects",
)


# Print post-install message
if 'install' in sys.argv or 'develop' in sys.argv:
    print("\n" + "="*70)
    print("GRF Installation Complete!")
    print("="*70)
    
    if CYTHON_AVAILABLE and ('build_ext' in sys.argv or '--with-cython' in sys.argv):
        print("\n[OK] Cython extensions compiled successfully")
        print("\nYou can use:")
        print("  from grf import CausalForestCython  # Fastest (matches econml)")
        print("  from grf import CausalForest  # Numba-accelerated")
    else:
        print("\n[OK] Numba-accelerated version ready")
        print("\nYou can use:")
        print("  from grf import CausalForest  # Numba (2-4x faster than pure Python)")
        print("\nFor maximum speed (match econml), install Cython extensions:")
        print("  python setup.py build_ext --inplace")
    
    print("\n" + "="*70 + "\n")