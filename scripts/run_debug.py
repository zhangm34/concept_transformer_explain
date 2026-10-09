"""Run the repository's one-batch debug workflow with its documented debug test fix and save its output."""
import json
import argparse
import ast
import re
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from time import perf_counter

argparse.ArgumentParser(description=__doc__).parse_args()

PROJECT_DIR = Path(__file__).resolve().parents[1]
RUN_DIR = PROJECT_DIR / 'training_runs' / 'debug'
RUN_DIR.mkdir(parents=True, exist_ok=True)
command = [
    sys.executable, str(PROJECT_DIR / 'concept_transformer' / 'cvit_cub.py'),
    '--data_dir', str(PROJECT_DIR / 'CUB-200-2011' / 'CUB_200_2011'),
    '--debug', '--batch_size', '2', '--num_workers', '0',
]
env = os.environ.copy()
env['TORCH_HOME'] = str(PROJECT_DIR / '.cache' / 'torch')
env['MPLCONFIGDIR'] = str(PROJECT_DIR / '.cache' / 'matplotlib')
env['PYTHONUNBUFFERED'] = '1'
import certifi
env['SSL_CERT_FILE'] = certifi.where()
record = {
    'started_at_utc': datetime.now(timezone.utc).isoformat(),
    'command': command,
    'working_directory': str(RUN_DIR),
    'torch_home': env['TORCH_HOME'],
    'log_file': str(RUN_DIR / 'run.log'),
    'repository_commit': json.loads((PROJECT_DIR / 'upstream.json').read_text())['commit'],
    'mode': 'repository --debug (fast_dev_run), with debug-only in-memory test fix',
    'batch_size': 2,
    'debug_test_fix': 'trainer.test(model=model, ckpt_path=None) if args.debug else trainer.test()',
    'entrypoint_sha256': hashlib.sha256((PROJECT_DIR / 'concept_transformer/cvit_cub.py').read_bytes()).hexdigest(),
    'num_workers': 0,
}
print('Starting repository CLI debug run; output: training_runs/debug/run.log', flush=True)
start = perf_counter()
with (RUN_DIR / 'run.log').open('w') as log:
    process = subprocess.Popen(command, cwd=str(RUN_DIR), env=env, stdout=log, stderr=subprocess.STDOUT)
    returncode = process.wait()
record.update({
    'returncode': returncode,
    'success': returncode == 0,
    'elapsed_seconds': perf_counter() - start,
    'finished_at_utc': datetime.now(timezone.utc).isoformat(),
})
if returncode == 0:
    match = re.search(r'DATALOADER:0 TEST RESULTS\s*(\{.*?\})',
                      (RUN_DIR / 'run.log').read_text(), flags=re.DOTALL)
    if match:
        record['test_metrics'] = ast.literal_eval(match.group(1))
(RUN_DIR / 'code.diff').write_text((PROJECT_DIR / 'patches/debug-test-no-checkpoint.patch').read_text())
(RUN_DIR / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
print(f'Debug run exit code: {returncode}; elapsed: {record["elapsed_seconds"]:.1f} seconds', flush=True)
sys.exit(returncode)
