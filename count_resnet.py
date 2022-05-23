from pathlib import Path

rest_set_path = Path('/data/rest_set')

for label in rest_set_path.iterdir():
    label_files = [f for f in label.glob('*')]
    print(f'There are {len(label_files)} samples in label {label.stem}')