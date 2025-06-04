import os
import json
deepspeed_config_path = os.getenv("DEEPSPEED_CONFIG", None)
if not deepspeed_config_path:
    print(False)


try:
    with open(deepspeed_config_path, "r") as f:
        config = json.load(f)

    zero_optimization = config.get("zero_optimization", {})
    print(zero_optimization.get("stage", 0) == 3)
except Exception as e:
    print(f"[WARNING] Could not parse DeepSpeed config: {e}")
    
