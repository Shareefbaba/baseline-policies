import sys
import importlib.util
from pathlib import Path

sys.path.append(str(Path.home() / "baseline_policies"))

spec = importlib.util.spec_from_file_location(
    "train_bc",
    Path.home() / "baseline_policies/scripts/train_bc.py"
)
train_bc_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_bc_module)
train = train_bc_module.train

FRANKA_TASKS = [
    "pick_cube",
    "open_drawer",
    "place",
    "push",
    "reach",
]

results = {}

for task in FRANKA_TASKS:
    print(f"\n{'='*40}")
    print(f"Training: franka/{task}")
    print(f"{'='*40}")

    try:
        train(
            robot="franka",
            task=task,
            epochs=500,
            lr=1e-4,
            batch_size=256,
            save_dir=f"checkpoints/franka_{task}_bc",
        )
        results[task] = "✅ done"
    except Exception as e:
        results[task] = f"❌ failed: {e}"
        print(f"FAILED: {e}")

print(f"\n{'='*40}")
print("SUMMARY")
print(f"{'='*40}")
for task, status in results.items():
    print(f"  franka/{task}: {status}")
