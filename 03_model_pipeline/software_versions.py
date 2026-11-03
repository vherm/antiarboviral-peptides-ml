from pathlib import Path
import platform
import sys

packages = [
    "numpy",
    "pandas",
    "sklearn",
    "scipy",
    "xgboost",
    "matplotlib",
    "openpyxl",
    "fpdf",
    "shap",
    "joblib",
]

output_file = Path("software_versions.txt")

lines = []
lines.append("Software versions")
lines.append("=================")
lines.append(f"Python: {sys.version.split()[0]}")
lines.append(f"Platform: {platform.platform()}")
lines.append("")

for package_name in packages:
    try:
        package = __import__(package_name)
        version = getattr(package, "__version__", "version not available")
        lines.append(f"{package_name}: {version}")
    except Exception as error:
        lines.append(f"{package_name}: not available ({error})")

output_file.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
print(f"\n[OK] Versions saved to: {output_file}")
