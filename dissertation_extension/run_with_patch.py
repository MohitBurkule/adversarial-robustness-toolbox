import os
import sys
import runpy

# Ensure dissertation_extension/ is in sys.path so patch_dataset is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import patch_dataset to monkey-patch torchvision datasets
import patch_dataset

if len(sys.argv) < 2:
    print("Usage: python run_with_patch.py <script_path> [args...]")
    sys.exit(1)

target_script = sys.argv[1]
# Reconstruct sys.argv so the target script sees itself as sys.argv[0]
sys.argv = sys.argv[1:]

print(f"[*] Executing script: {target_script}")
runpy.run_path(target_script, run_name="__main__")
