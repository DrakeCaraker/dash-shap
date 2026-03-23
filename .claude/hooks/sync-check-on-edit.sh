#!/usr/bin/env bash
export REPO="$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
f=$(jq -r '.tool_input.file_path // ""')
basename "$f" | grep -qE '^(run_experiments.*\.py|config\.py)$' || exit 0
echo "Checking PAPER_CONFIG sync..."
python3 << 'PYEOF'
import re, sys, os
REPO = os.environ.get('REPO', '.')

def parse_config(text):
    return {
        'M': int(re.search(r'"M":\s*(\d+)', text).group(1)),
        'K': int(re.search(r'"K":\s*(\d+)', text).group(1)),
        'N_REPS': int(re.search(r'"N_REPS":\s*(\d+)', text).group(1)),
        'EPSILON': float(re.search(r'"EPSILON":\s*([\d.]+)', text).group(1)),
        'DELTA': float(re.search(r'"DELTA":\s*([\d.]+)', text).group(1)),
        'SEED': int(re.search(r'SEED\s*(?::\s*int\s*)?=\s*(\d+)', text).group(1)),
    }

# Canonical source of truth + sequential runner (which still has a local copy)
files = [
    os.path.join(REPO, 'dash_shap', 'config.py'),
    os.path.join(REPO, 'run_experiments.py'),
]
configs = {}
for path in files:
    try:
        configs[path] = parse_config(open(path).read())
    except Exception as e:
        print(f'Could not parse {path}: {e}')
        sys.exit(1)

ref_path, ref = list(configs.items())[0]
errors = 0
for path, vals in list(configs.items())[1:]:
    for key in ref:
        if vals.get(key) != ref[key]:
            print(f'MISMATCH {key}: config.py={ref[key]}, run_experiments.py={vals.get(key)}')
            errors += 1
if errors == 0:
    print('PAPER_CONFIG consistent.')
sys.exit(errors)
PYEOF
