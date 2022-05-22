from pathlib import Path
import random
random.seed(42)

def fill_sub_dir(sub_dir, file_subset):
    """This function copies files from the `train_all` to a `<sub_dir>`
    A more efficient solution would be to use "symbolic links" (see https://kb.iu.edu/d/abbe)
    but for simplicity hard copies is used instead.
    """
    for file in file_subset:
        file_path = Path.cwd() / sub_dir / file.name        
        file_path.symlink_to(file)

if __name__ == '__main__':

    crops_200_path = Path('/data/small_crops_200')
    crops_1000_path = Path('/data/small_crops_1000')
    crops_path = Path('/data/crops')

    # File names in training sets:
    fnames_200 = [f.name for f in crops_200_path.glob('*/*.tiff')]
    fnames_1000 = [f.name for f in crops_1000_path.glob('*/*.tiff')]

    assert len(fnames_200)==(200*11), f'Number of files in small_crops_200 is {len(fnames_200)}, but should be {11*200}'
    assert len(fnames_1000)==(1000*11), f'Number of files in small_crops_200 is {len(fnames_1000)}, but should be {11*1000}'

    all_training_files = fnames_200+fnames_1000

    allowed_dict = {}
    # Iterate through all label directories and list all files that are not in the training set:
    for subdir in crops_path.iterdir():
        allowed_dict[subdir.name] = [f for f in subdir.glob('*.tiff') if f.name not in all_training_files]

    test_path = Path('/data/test_crops')
    if not test_path.exists():
        test_path.mkdir()
    else:
        for subdir in test_path.iterdir():
            for item in subdir.iterdir():
                item.unlink()
            subdir.rmdir()
        test_path.rmdir()


    for subdir in allowed_dict:
        new_subdir = test_path / subdir
        if new_subdir.exists():
            for item in new_subdir.iterdir():
                item.unlink()
            new_subdir.rmdir()

        new_subdir.mkdir()
        
        # Sample 1000 random files from each class:
        small_subset = random.sample(allowed_dict[subdir],k=1000)
        fill_sub_dir(new_subdir, small_subset)
        print('Done with subdir {}'.format(subdir))

    for subdir in test_path.iterdir():
        print('Number of samples in class {}: {}'.format(subdir.stem,len([item for item in subdir.glob('*')])))

    # Asserting that no test files are in training set:
    test_file_names = [f.name for f in test_path.glob('*/*.tiff')]
    for f in test_file_names:
        if f in all_training_files:
            print('File {} is in both test and training set!'.format(f))

    with open('test_files.txt','w') as f:
        for file in test_file_names:
            _ = file + '\n'
            f.write(_)

    





