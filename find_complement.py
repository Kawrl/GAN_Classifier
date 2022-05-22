from pathlib import Path

test_dir = Path('/data/test_crops')
train_dir_1000 = Path('/data/small_crops_1000')
train_dir_200 = Path('/data/small_crops_200')

train_files_1000 = [f.name for f in train_dir_1000.glob('*/*')]
train_files_200 = [f.name for f in train_dir_200.glob('*/*')]
train_files = train_files_1000+train_files_200
test_dir_names = [f.name for f in test_dir.glob('*/*')]

diff = list(set(train_files) - set(test_dir_names))
print(diff)
assert sorted(diff) == sorted(train_files), "Something is wrong. This assumes there is leakage from test to train."

test_dir = Path('/data/test_crops')
crops_dir = Path('/data/crops')

all_file_names = [f.name for f in crops_dir.glob('*/*.tiff')]
rest_files = [f for f in crops_dir.glob('*/*.tiff') if f.name not in test_dir_names]
rest_files_names = [f.name for f in rest_files]
set_names = list(set(all_file_names) - set(test_dir_names))

assert sorted(set_names)==sorted(rest_files_names), "Set method is not the same as iter method!"

