from pathlib import Path
from tqdm import tqdm
from time import time
import torch
from sklearn.decomposition import PCA
import pickle
from cropsdata import create_dataset


def create_img_dict(dataset):
    '''
    Creates a dictionary of flattened images for each image in their respective classes.
    '''
    num_classes = len(dataset.label_dict.keys())

    img_dict = {i:[] for i in range(num_classes)}

    for idx, (img,label,idx) in enumerate(tqdm(dataset)):
        flattened_img = img[0].flatten() # Only the first channel (since all channels are the same)
        
        # Store tuples of flattened images and their index in dataset in order to keep track of them.
        img_dict[label].append((flattened_img, idx))

    return img_dict

def create_norm_dict(dataset,img_dict, num_components=682):
    '''
    Computes a dictionary with norms of projection vectors(?) for each data point per class,
    from the subspace that spans the given number of principal components.
    If norm_dct_pickle is set to True, the function attempts to load the pickle data.
    '''

    pca_dct = {}    
    num_classes = len(dataset.label_dict.keys())

    for label in range(num_classes):
        start = time()
        imgs, idxs = zip(*img_dict[label])
        stacked_imgs = torch.stack(imgs)
        pca = PCA(num_components)
        pca.fit(stacked_imgs)
        print(f'Total variance explained by {num_components} principal components: {pca.explained_variance_ratio_.sum()}')
        
        print(f"Done with class {label} in {time()-start:.3f}s.")
        pca_dct[label]=pca

    # Saving pca_dct as pickle:
    with open('pca_dct_full_set.pickle', 'wb') as handle:
        pickle.dump(pca_dct, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return pca_dct

# Put the training and validation data in the respective folders
def fill_sub_dir(sub_dir, file_subset):
    """This function copies files from the `train_all` to a `<sub_dir>`
    A more efficient solution would be to use "symbolic links" (see https://kb.iu.edu/d/abbe)
    but for simplicity hard copies is used instead.
    """
    for file in file_subset:
        file_path = sub_dir / file.name        
        file_path.symlink_to(file)


if __name__ == '__main__':

    rest_set_path = Path('/data/rest_set')
    if not rest_set_path.exists():
        rest_set_path.mkdir()

    crops_path = Path('/data/crops')
    test_path = Path('/data/test_crops')

    rest_file_dict = {}

    # Iterate through all files in the full dataset to select files that are NOT in the test set.
    for label in crops_path.iterdir():
        all_files = list(label.glob('*.tiff'))
        test_files = [f.name for f in list(test_path.glob(f'{label.stem}/*.tiff'))]
        rest_files = [f for f in all_files if f.name not in test_files]

        new_subdir = rest_set_path / label.stem
        if new_subdir.exists():
            for item in new_subdir.iterdir():
                item.unlink()
            new_subdir.rmdir()

        new_subdir.mkdir()
        
        fill_sub_dir(new_subdir, rest_files)
        print('Done with subdir {}'.format(label.stem))

    for subdir in rest_set_path.iterdir():
        print('Number of samples in class {}: {}'.format(subdir.stem,len([item for item in subdir.glob('*')])))

    assert len(list(set(list(rest_set_path.glob('*/*.tiff'))) & set(list(test_path.glob('*/*.tiff'))))) == 0, \
        f"Intersection of train and test files: {list(set(list(rest_set_path.glob('*/*.tiff'))) & set(list(test_path.glob('*/*.tiff'))))}"

    # Create rest set:
    rest_data = create_dataset(rest_set_path)

    # Generate dictinoary of all flattened files per class in the rest set:
    rest_img_dict = create_img_dict(rest_data)

    # ... and then generate a subspace based on the full set (excluding the test set)
    rest_pca_dct = create_norm_dict(rest_data,rest_img_dict, num_components=682)

