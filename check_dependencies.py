"""
Script to check if all dependencies are installed correctly
"""

import sys

def check_dependency(name, package=None):
    """Check if a package is installed."""
    if package is None:
        package = name

    try:
        __import__(package)
        print(f"✓ {name} is installed")
        return True
    except ImportError:
        print(f"✗ {name} is NOT installed")
        return False


def main():
    print("="*60)
    print("A-LSA Dependency Check")
    print("="*60)

    dependencies = [
        ("NumPy", "numpy"),
        ("SciPy", "scipy"),
        ("scikit-learn", "sklearn"),
        ("NLTK", "nltk"),
        ("pandas", "pandas"),
        ("tabulate", "tabulate"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("tqdm", "tqdm"),
        ("psutil", "psutil"),
    ]

    results = []

    print("\nChecking core dependencies:")
    print("-" * 60)

    for name, package in dependencies:
        results.append(check_dependency(name, package))

    # Check NLTK data
    print("\nChecking NLTK data:")
    print("-" * 60)
    try:
        import nltk
        try:
            nltk.data.find('corpora/stopwords')
            print("✓ NLTK stopwords data is downloaded")
            results.append(True)
        except LookupError:
            print("✗ NLTK stopwords data is NOT downloaded")
            print("  Run: python -c \"import nltk; nltk.download('stopwords')\"")
            results.append(False)
    except ImportError:
        print("✗ NLTK is not installed, cannot check data")
        results.append(False)

    # Optional dependencies
    print("\nChecking optional dependencies:")
    print("-" * 60)

    optional = [
        ("Jupyter", "jupyter"),
        ("ipykernel", "ipykernel"),
    ]

    for name, package in optional:
        try:
            __import__(package)
            print(f"✓ {name} is installed (optional)")
        except ImportError:
            print(f"○ {name} is NOT installed (optional, for notebooks)")

    # Summary
    print("\n" + "="*60)
    total = len(results)
    passed = sum(results)

    if passed == total:
        print(f"✓ ALL CHECKS PASSED ({passed}/{total})")
        print("="*60)
        print("\nYou can now run the experiments:")
        print("  python test_simple.py")
        print("  python experiments/run_newsgroups.py")
        print("  python experiments/run_sms_spam.py")
        print("  python experiments/run_imdb.py")
        return 0
    else:
        print(f"✗ SOME CHECKS FAILED ({passed}/{total})")
        print("="*60)
        print("\nTo install missing dependencies:")
        print("  pip install -r requirements.txt")
        print("  python -c \"import nltk; nltk.download('stopwords')\"")
        return 1


if __name__ == "__main__":
    sys.exit(main())
