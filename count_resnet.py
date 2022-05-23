from pathlib import Path

rest_set_path = Path('/data/rest_set')
full_set_path = Path('/data/crops')
total_=0
for label in rest_set_path.iterdir():
    label_files = [f for f in label.glob('*')]
    print(f'There are {len(label_files)} samples in label {label.stem}')
    total_+=len(label_files)

print(f'In total: {total_} files')

all_rest_files = rest_set_path.glob('*/*.tiff')
all_files = full_set_path.glob('*/*.tiff')
assert len(list(all_rest_files))==total_, "Something wrong with file counting?"

all_rest_file_names = [f.name for f in all_rest_files]
full_file_names = [f.name for f in all_files]

assert len(list(set(all_rest_file_names) & set(full_file_names))) == 0, \
        f"Intersection of train and test files: {list(set(all_rest_file_names) & set(full_file_names))}"

