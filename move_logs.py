from pathlib import Path
import shutil

log_dir = Path('/data/saved_classifier_logs')
new_log_dir = log_dir / 'logs'
if not new_log_dir.exists():
    new_log_dir.mkdir()

log_files = list(log_dir.glob('*.log'))

for file in log_files:
    new_file = new_log_dir / file.name
    shutil.copy(file, new_file)
