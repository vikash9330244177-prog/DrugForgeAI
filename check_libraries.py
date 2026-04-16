"""
DrugForge AI library checker

This script checks whether commonly used Python dependencies are installed
and prints their import status and version information.
"""

LIBRARIES = [
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "sklearn",
    "torch",
    "dgl",
    "rdkit",
    "openmm",
    "mdtraj",
    "requests",
    "flask",
    "joblib",
    "umap",
    "Bio",
    "admet_ai",
    "deepchem",
]

print("\nChecking installed Python libraries...\n")

for lib in LIBRARIES:
    try:
        module = __import__(lib)
        version = getattr(module, "__version__", "version info not available")
        print(f"[OK] {lib} is installed. Version: {version}")
    except ImportError as exc:
        print(f"[MISSING] {lib} is not installed or import failed.")
        print(f"          Error: {exc}")
    except Exception as exc:
        print(f"[WARNING] {lib} import produced an unexpected error.")
        print(f"          Error: {exc}")

print("\nLibrary check complete.")