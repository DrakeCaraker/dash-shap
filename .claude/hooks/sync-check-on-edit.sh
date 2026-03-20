#!/usr/bin/env bash
REPO=/Users/drake.caraker/ds_projects/dash-shap
f=$(jq -r '.tool_input.file_path // ""')
basename "$f" | grep -qE '^run_experiments.*\.py$' || exit 0
echo "Checking PAPER_CONFIG sync..."
python3 -c "
import re, sys
files = ['$REPO/run_experiments.py', '$REPO/run_experiments_parallel.py']
configs = {}
for path in files:
    try:
        content = open(path).read()
        configs[path] = {
            'M': int(re.search(r'[\"\\x27]M[\"\\x27]:\\s*(\\d+)', content).group(1)),
            'K': int(re.search(r'[\"\\x27]K[\"\\x27]:\\s*(\\d+)', content).group(1)),
            'N_REPS': int(re.search(r'[\"\\x27]N_REPS[\"\\x27]:\\s*(\\d+)', content).group(1)),
            'EPSILON': float(re.search(r'[\"\\x27]EPSILON[\"\\x27]:\\s*([\\d.]+)', content).group(1)),
            'DELTA': float(re.search(r'[\"\\x27]DELTA[\"\\x27]:\\s*([\\d.]+)', content).group(1)),
            'SEED': int(re.search(r'SEED\\s*=\\s*(\\d+)', content).group(1)),
        }
    except Exception as e:
        print(f'Could not parse {path}: {e}')
        sys.exit(1)
ref_path, ref = list(configs.items())[0]
errors = 0
for path, vals in list(configs.items())[1:]:
    for key in ref:
        if vals.get(key) != ref[key]:
            print(f'MISMATCH {key}: run_experiments.py={ref[key]}, parallel={vals.get(key)}')
            errors += 1
if errors == 0:
    print('PAPER_CONFIG consistent.')
sys.exit(errors)
" 2>&1 || true
