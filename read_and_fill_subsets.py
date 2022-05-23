from pathlib import Path

def read_file_file(file):
    '''
    Reads file containing list of file names. Returns list
    of all file names.
    '''
    file_list = []
    with open(file,'r') as f:
        file_list += f.read().split()
    return file_list

def fill_sub_dir(sub_dir, file_subset):
    """This function copies files from the `train_all` to a `<sub_dir>`
    A more efficient solution would be to use "symbolic links" (see https://kb.iu.edu/d/abbe)
    but for simplicity hard copies is used instead.
    """
    for file in file_subset:
        file_path = Path.cwd() / sub_dir / file.name        
        file_path.symlink_to(file)

def subdir_check(subdir):
    if not subdir.exists():
        subdir.mkdir()

if __name__ == '__main__':

    files_200 = read_file_file('files_small_crops_200_AIXIA.txt')
    files_1000 = read_file_file('files_small_crops_1000_AIXIA.txt')

    print('Examples of files_200:')
    for f in files_200[:5]:
        print(f)


    new_crop_subset_path_200 = Path('/data/small_crops_200')
    subdir_check(new_crop_subset_path_200)

    new_crop_subset_path_1000 = Path('/data/small_crops_1000')
    subdir_check(new_crop_subset_path_1000)

    crops_path = Path('/data/crops')

    for subdir in crops_path.iterdir():
        new_subdir_200 = new_crop_subset_path_200 / subdir.stem
        new_subdir_1000 = new_crop_subset_path_1000 / subdir.stem

        subdir_check(new_subdir_200)
        subdir_check(new_subdir_1000)
        all_filenames = list(subdir.glob('*.tiff'))
        
        small_subset_200 = [f for f in all_filenames if f.name in files_200]
        small_subset_1000 = [f for f in all_filenames if f.name in files_1000]

        fill_sub_dir(new_subdir_200, small_subset_200)
        fill_sub_dir(new_subdir_1000, small_subset_200)
        print('Done with subdir {}'.format(subdir))

    print('Small_crops_200:')
    for subdir in new_crop_subset_path_200.iterdir():
        print('Number of samples in class {}: {}'.format(subdir.stem,len([item for item in subdir.glob('*')])))
    print('Small_crops_1000:')
    for subdir in new_crop_subset_path_1000.iterdir():
        print('Number of samples in class {}: {}'.format(subdir.stem,len([item for item in subdir.glob('*')])))

