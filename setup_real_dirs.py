# Creating dataset where all images has heigh < 256 and width < 128, and only belongs to label 0,1,2,3,4

import random
from argparse import ArgumentParser
from pathlib import Path
random.seed(42)

# Put the training and validation data in the respective folders
def fill_sub_dir(sub_dir, file_subset):
    """This function copies files from the `train_all` to a `<sub_dir>`
    A more efficient solution would be to use "symbolic links" (see https://kb.iu.edu/d/abbe)
    but for simplicity hard copies is used instead.
    """
    for file in file_subset:
        file_path = Path.cwd() / sub_dir / file.name        
        file_path.symlink_to(file)

def load_parser():
    parser = ArgumentParser(add_help=True)
    parser.add_argument("--num_samples", type=int, default=1)
    args = parser.parse_args()    

    return args

if __name__ == '__main__':
    args = load_parser()
    #save_dir = args.save_dir
    num_samples = args.num_samples

    new_crop_subset_path = Path('/data/small_crops')
    if not new_crop_subset_path.exists():
        new_crop_subset_path.mkdir()

    crops_path = Path('/data/crops')

    for subdir in crops_path.iterdir():
        new_subdir = new_crop_subset_path / subdir.stem
        if new_subdir.exists():
            for item in new_subdir.iterdir():
                item.unlink()
            new_subdir.rmdir()

        new_subdir.mkdir()
        all_filenames = list(subdir.glob('*.tiff'))
        
        # Sample 1000 random files from each class:
        small_subset = random.sample(all_filenames,k=num_samples)
        fill_sub_dir(new_subdir, small_subset)
        print('Done with subdir {}'.format(subdir))

    for subdir in new_crop_subset_path.iterdir():
        print('Number of samples in class {}: {}'.format(subdir.stem,len([item for item in subdir.glob('*')])))