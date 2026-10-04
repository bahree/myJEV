"""Optional W&B metrics. Local JSONL remains authoritative; default is disabled."""
import os
from pathlib import Path


def start_run(output, config, *, mode=None, historical=False):
    mode = mode or os.environ.get('MYJEV_TRACKING_MODE', 'disabled')
    if mode not in {'disabled', 'offline', 'online'}:
        raise ValueError('tracking mode must be disabled, offline, or online')
    if mode == 'disabled':
        return None
    project = os.environ.get('MYJEV_WANDB_PROJECT')
    if mode == 'online' and not project:
        raise ValueError('online tracking requires MYJEV_WANDB_PROJECT')
    import wandb
    directory = Path(output)/'tracking'
    directory.mkdir(parents=True, exist_ok=True)
    return wandb.init(project=project or 'myjev', entity=os.environ.get('MYJEV_WANDB_ENTITY'),
                      name=Path(output).name, dir=str(directory), mode=mode,
                      config=config, save_code=False,
                      tags=['historical-import' if historical else 'live-training'],
                      settings=wandb.Settings(console='off', disable_git=True,
                                              x_disable_stats=historical))
