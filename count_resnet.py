from pathlib import Path

rest_set_path = Path('/data/rest_set')
test_set_path = Path('/data/test_crops')
total_=0
for label in rest_set_path.iterdir():
    label_files = [f for f in label.glob('*')]
    print(f'There are {len(label_files)} samples in label {label.stem}')
    total_+=len(label_files)

print(f'In total: {total_} files')

all_rest_files = list(rest_set_path.glob('*/*'))
all_files = list(test_set_path.glob('*/*'))
assert len(all_rest_files)==total_, "Something wrong with file counting?"

all_rest_file_names = [f.name for f in all_rest_files]
full_file_names = [f.name for f in all_files]
print(len(all_rest_file_names))
print(len(full_file_names))

assert len(list(set(all_rest_file_names) & set(full_file_names))) == 0, \
        f"Intersection of train and test files: {list(set(all_rest_file_names) & set(full_file_names))}"

for f in all_rest_file_names:
    if f in full_file_names:
        print(f)
