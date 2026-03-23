import os

def initialize_minsky_structure():
    # 1. Define the required directory tree
    folders = [
        "config",
        "data/results",
        "models",
        "src/environment",
        "src/evaluation",
        "src/policies",
        "src/primitives",
        "src/training"
    ]

    print("--- Initializing Minsky Project Structure ---")

    # 2. Create Folders
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"[CREATED] Directory: {folder}")
        else:
            print(f"[EXISTS]  Directory: {folder}")

    # 3. Create __init__.py files in all src subfolders
    # This allows: from src.environment import ...
    src_path = "src"
    for root, dirs, files in os.walk(src_path):
        init_file = os.path.join(root, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                pass # Create empty file
            print(f"[CREATED] Package Init: {init_file}")

    # 4. Critical File Renaming (The YAML Fix)
    old_yaml = "config/hr1_config.yoml.txt"
    new_yaml = "config/hrl_config.yaml"
    if os.path.exists(old_yaml):
        os.rename(old_yaml, new_yaml)
        print(f"[RENAMED] {old_yaml} -> {new_yaml}")

    print("\n--- Setup Complete. You are ready to train! ---")

if __name__ == "__main__":
    initialize_minsky_structure()