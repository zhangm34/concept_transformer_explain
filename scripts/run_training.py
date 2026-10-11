"""Run the repository's regular training workflow with a timing-only callback."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback

PROJECT_DIR = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-dir', type=Path, required=True)
parser.add_argument('--max-epochs', type=int, default=50)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument('--require-cuda', action='store_true', help='Fail unless an allocated CUDA GPU is available.')
args = parser.parse_args()
if args.batch_size < 2 or args.num_workers < 0 or args.max_epochs < 1:
    parser.error('Use batch-size >= 2, num-workers >= 0, and max-epochs >= 1.')
if args.require_cuda:
    import torch
    if not torch.cuda.is_available():
        parser.error('--require-cuda requires a visible CUDA GPU; submit inside a GPU allocation.')
RUN_DIR = args.run_dir.resolve()
RUN_DIR.mkdir(parents=True, exist_ok=True)
entry = PROJECT_DIR / 'concept_transformer' / 'cvit_cub.py'
data_dir = PROJECT_DIR / 'CUB-200-2011' / 'CUB_200_2011'
cli_args = ['--data_dir', str(data_dir), '--batch_size', str(args.batch_size), '--num_workers', str(args.num_workers),
            '--max_epochs', str(args.max_epochs)]
command = [sys.executable, str(entry)] + cli_args
os.environ['TORCH_HOME'] = str(PROJECT_DIR / '.cache' / 'torch')
os.environ['MPLCONFIGDIR'] = str(PROJECT_DIR / '.cache' / 'matplotlib')
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.chdir(RUN_DIR)
sys.path.insert(0, str(entry.parent))

record_path = RUN_DIR / 'run.json'
record = {
    'started_at_utc': datetime.now(timezone.utc).isoformat(),
    'pid': os.getpid(), 'status': 'initializing', 'debug': False,
    'command': command, 'working_directory': str(RUN_DIR),
    'runner': str(PROJECT_DIR / 'scripts' / 'run_training.py'),
    'runner_sha256': hashlib.sha256((PROJECT_DIR / 'scripts' / 'run_training.py').read_bytes()).hexdigest(),
    'repository_commit': json.loads((PROJECT_DIR / 'upstream.json').read_text())['commit'],
    'entrypoint_sha256': hashlib.sha256(entry.read_bytes()).hexdigest(),
    'max_epochs': args.max_epochs, 'batch_size': args.batch_size, 'num_workers': args.num_workers,
    'timing_file': str(RUN_DIR / 'batch_timings.jsonl'),
    'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
    'slurm_partition': os.environ.get('SLURM_JOB_PARTITION'),
    'slurm_cpus_per_task': os.environ.get('SLURM_CPUS_PER_TASK'),
    'require_cuda': args.require_cuda,
    'instrumentation': 'Timing callback; original model, optimizer, loss, and CLI defaults with Lightning 1.9 native AMP trainer.',
}

def save_record():
    temp = record_path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(record, indent=2) + '\n')
    temp.replace(record_path)

save_record()
(RUN_DIR / 'code.diff').write_text(''.join(path.read_text() for path in sorted((PROJECT_DIR / 'patches').glob('*.patch'))))
print('Starting regular training with repository CLI defaults, native AMP, and a timing callback.', flush=True)
start = time.perf_counter()
try:
    import pytorch_lightning as pl
    import ctc.ctc_model as training

    class BatchTiming(pl.Callback):
        def __init__(self):
            super().__init__()
            self.last_train_end = None
            self.phase_start = None
            self.train_start = None

        def write(self, row):
            row['recorded_at_utc'] = datetime.now(timezone.utc).isoformat()
            with (RUN_DIR / 'batch_timings.jsonl').open('a') as handle:
                handle.write(json.dumps(row) + '\n')

        def on_train_start(self, trainer, pl_module):
            record['status'] = 'training'
            record['device'] = str(pl_module.device)
            record['train_batches_per_epoch'] = int(trainer.num_training_batches)
            record['validation_batches_per_epoch'] = [int(x) for x in trainer.num_val_batches]
            record['training_started_at_utc'] = datetime.now(timezone.utc).isoformat()
            save_record()

        def on_train_epoch_start(self, trainer, pl_module):
            self.last_train_end = None

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
            self.train_start = time.perf_counter()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            now = time.perf_counter()
            loss = outputs.get('loss') if isinstance(outputs, dict) else None
            self.write({'phase': 'train', 'epoch': int(trainer.current_epoch),
                        'batch_index': int(batch_idx), 'global_step': int(trainer.global_step),
                        'compute_seconds': now - self.train_start,
                        'wall_seconds': now - self.last_train_end if self.last_train_end is not None else now - self.train_start,
                        'loss': float(loss.detach().cpu()) if loss is not None else None,
                        'learning_rate': float(trainer.optimizers[0].param_groups[0]['lr'])})
            self.last_train_end = now

        def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
            self.phase_start = time.perf_counter()

        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            self.write({'phase': 'sanity_validation' if trainer.sanity_checking else 'validation',
                        'epoch': int(trainer.current_epoch), 'batch_index': int(batch_idx),
                        'compute_seconds': time.perf_counter() - self.phase_start})

        def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
            self.phase_start = time.perf_counter()

        def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            self.write({'phase': 'test', 'batch_index': int(batch_idx),
                        'compute_seconds': time.perf_counter() - self.phase_start})

    original_get_trainer = training.get_trainer

    def timed_get_trainer(max_epochs, logger, callbacks, amp=False, debug=False):
        return original_get_trainer(max_epochs, logger, list(callbacks) + [BatchTiming()], amp=amp, debug=debug)

    training.get_trainer = timed_get_trainer
    sys.argv = [str(entry)] + cli_args
    runpy.run_path(str(entry), run_name='__main__')
except BaseException as error:
    record['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
    record['error'] = f'{type(error).__name__}: {error}'
    traceback.print_exc()
    record['elapsed_seconds'] = time.perf_counter() - start
    record['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    save_record()
    raise
else:
    record['status'] = 'complete'
    record['elapsed_seconds'] = time.perf_counter() - start
    record['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    save_record()
