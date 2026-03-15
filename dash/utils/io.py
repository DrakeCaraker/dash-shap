"""I/O utilities for saving experiment results."""
import json
import numpy as np


def save_json(data, path):
    """Save results dict to JSON, converting numpy types to native Python."""
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    with open(path, 'w') as f:
        json.dump(convert(data), f, indent=2)
