from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
import random

RANDOM_STATE = 42
random.seed(RANDOM_STATE)

def fill_sub_dir(sub_dir, file_subset, new_dir_path,fake=False):
    """This function copies files from the `train_all` to a `<sub_dir>`
    A more efficient solution would be to use "symbolic links" (see https://kb.iu.edu/d/abbe)
    but for simplicity hard copies is used instead.
    """
    sub_dir_path = new_dir_path / sub_dir
    if not sub_dir_path.exists():
        sub_dir_path.mkdir()

    for file in file_subset:
        fname = file.name
        if fake:
            fname = 'fake_' + file.name
        new_file = sub_dir_path / fname
        
        # In case of file already existing:
        if new_file.exists():
            fname = 'fake_2_' + file.name
            new_file = sub_dir_path / fname

        new_file.symlink_to(file)

def add_fake_samples(img_dir,fake_dir,add_samples_per_class):
    aug_name = 'augmented_'+str(add_samples_per_class)
    aug_path = Path.cwd() / aug_name
    

    
    if aug_path.exists():
        if aug_path.is_dir():
            shutil.rmtree(aug_path)
        else:
            print("Unknown item: {}, remove manually".format(aug_path))

    if not aug_path.exists():
        aug_path.mkdir()

    print('Adding all real samples:')
    for label_dir in img_dir.iterdir():
        if not label_dir.is_dir():
            continue
        label = label_dir.stem
        all_files = list(label_dir.glob('*'))
        
        # Moving train/val set:
        fill_sub_dir(label, all_files, aug_path)

    print(f'Adding {add_samples_per_class} of fake images per class:')
    for label_dir in fake_dir.iterdir():
        if not label_dir.is_dir():
            continue
        label = label_dir.stem
        all_files = list(label_dir.glob('*'))
        subset = random.sample(all_files, add_samples_per_class)
        
        # Moving train/val set:
        fill_sub_dir(label, subset, aug_path,fake=True)


    original_num_samples = len(list(img_dir.glob('*/*')))
    augmented_num_samples = len(list(aug_path.glob('*/*')))

    print(f'There were originally {original_num_samples} number of samples in the training set.',
          f'\nNumber of augmented training set is now {augmented_num_samples}')

    return aug_path

def add_pca_samples(img_dir,fake_dir,add_samples_per_class,path_dct):
    aug_name = 'augmented_norm_'+str(add_samples_per_class)
    aug_path = Path.cwd() / aug_name
    
    if aug_path.exists():
        if aug_path.is_dir():
            shutil.rmtree(aug_path)
        else:
            print("Unknown item: {}, remove manually".format(aug_path))

    if not aug_path.exists():
        aug_path.mkdir()

    print('Adding all real samples:')
    for label_dir in img_dir.iterdir():
        if not label_dir.is_dir():
            continue
        label = label_dir.stem
        all_files = list(label_dir.glob('*'))
        
        # Moving train/val set:
        fill_sub_dir(label, all_files, aug_path)

    print(f'Adding {add_samples_per_class} of fake images per class:')
    for label_dir in fake_dir.iterdir():
        if not label_dir.is_dir():
            continue
        label = label_dir.stem
        all_files = path_dct[int(label)]

        if add_samples_per_class > len(all_files):
            # In case we want to sample more imgs than available,
            # some images will be randomly upsampled.
            
            diff = add_samples_per_class-len(all_files)           
            
        else:
            diff = 0


        subset = random.sample(all_files, add_samples_per_class-diff)
        diff = min(diff, len(all_files))
        subset += random.sample(all_files, diff)
        
        # Moving train/val set:
        fill_sub_dir(label, subset, aug_path,fake=True)


    original_num_samples = len(list(img_dir.glob('*/*')))
    augmented_num_samples = len(list(aug_path.glob('*/*')))

    print(f'There were originally {original_num_samples} number of samples in the training set.',
          f'\nNumber of augmented training set is now {augmented_num_samples}')

    return aug_path

def add_pca_samples_over_under_90(img_dir,fake_dir,add_samples_per_class,path_dict_over90, path_dict_under90):
    aug_name = 'augmented_norm_'+str(add_samples_per_class)
    aug_path = Path.cwd() / aug_name
    if aug_path.exists():
        if aug_path.is_dir():
            shutil.rmtree(aug_path)
        else:
            print("Unknown item: {}, remove manually".format(aug_path))

    if not aug_path.exists():
        aug_path.mkdir()

    print('Adding all real samples:')
    for label_dir in img_dir.iterdir():
        if not label_dir.is_dir():
            continue
        label = label_dir.stem
        all_files = list(label_dir.glob('*'))
        
        # Moving train/val set:
        fill_sub_dir(label, all_files, aug_path)

    print(f'Adding {add_samples_per_class} of fake images per class:')
    for label_dir in fake_dir.iterdir():
        if not label_dir.is_dir():
            continue
        label = label_dir.stem
        outlier_files = path_dict_over90[int(label)]
        normal_files = path_dict_under90[int(label)]

        outlier_size = add_samples_per_class // 2
        normal_size = add_samples_per_class-outlier_size

        subset_outliers = random.sample(outlier_files, outlier_size)
        subset_normal = random.sample(normal_files, normal_size)
        
        subset = subset_outliers+subset_normal
        
        # Moving train/val set:
        fill_sub_dir(label, subset, aug_path,fake=True)


    original_num_samples = len(list(img_dir.glob('*/*')))
    augmented_num_samples = len(list(aug_path.glob('*/*')))

    print(f'There were originally {original_num_samples} number of samples in the training set.',
          f'\nNumber of augmented training set is now {augmented_num_samples}')

    return aug_path

