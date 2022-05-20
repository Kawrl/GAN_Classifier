from sklearn.decomposition import PCA

import torch
from time import time
from tqdm import tqdm
import numpy as np
from cropsdata import create_dataset
from pathlib import Path
import pickle
from matplotlib import pyplot as plt


#def compute_norm(vector: torch.tensor) -> torch.tensor:
#    norm = ((vector**2).sum(dim=-1)).sqrt()
#    return norm.numpy()

def create_img_dict(dataset):    
    num_classes = len(dataset.label_dict.keys())

    img_dict = {i:[] for i in range(num_classes)}

    for idx, (img,label,idx) in enumerate(tqdm(dataset)):
        flattened_img = img[0].flatten() # Only the first channel (since all channels are the same)
        
        # Store tuples of flattened images and their index in dataset in order to keep track of them.
        img_dict[label].append((flattened_img, idx))

    # Done in around 5 mins
    with open('img_dict.pickle', 'wb') as handle:
        pickle.dump(img_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return img_dict

def create_norm_dict(dataset,img_dict,norm_dct_pickle=False, num_components=682):
    '''
    Computes a dictionary with norms of projection vectors(?) for each data point per class,
    from the subspace that spans the given number of principal components.
    If norm_dct_pickle is set to True, the function attempts to load the pickle data.

    '''

    pca_dct = {}
    norm_dct = {}

    if not norm_dct_pickle:
        num_classes = len(dataset.label_dict.keys())      


        for label in range(num_classes):
            start = time()
            imgs, idxs = zip(*img_dict[label])
            stacked_imgs = torch.stack(imgs)
            num_components = min(num_components, len(imgs))
            num_components = 100
            pca = PCA(num_components)
            pca.fit(stacked_imgs)
            print(f'Total variance explained by {num_components} principal components: {pca.explained_variance_ratio_.sum()}')
            
            X_pca = pca.transform(stacked_imgs)
            X_proj = pca.inverse_transform(X_pca)

            reconstruct_diff = stacked_imgs - X_proj
            # Compute norms of error vectors:
            norm = np.linalg.norm(reconstruct_diff,axis=-1)
            # norm = compute_norm(reconstruct_diff)
            norm_dct[label] = list(zip(norm,idxs))
            print(f"Done with class {label} in {time()-start:.3f}s.")
            pca_dct[label]=pca

    return norm_dct, pca_dct

def sort_norm_dict(norm_dict):
    '''
    Sorts a given dictionary of computed vector norms.
    '''

    sorted_norms_idxs = {}
    for label in norm_dict:
        sorted_norms_idxs[label] = sorted(norm_dict[label],key=lambda x:x[0],reverse=True)
    
    return sorted_norms_idxs

def plot_norm_hist(norm_dct,pca_dct,num_components):

    sorted_norms_idxs = sort_norm_dict(norm_dct)

    nrow = 4
    ncol = 3
    h_count=0

    cdict = {i:str(i) for i in range(11)}
    cdict[10]='W'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    fig, ax = plt.subplots(nrow,ncol,figsize=(16,12),sharex=True)
    for i, ax in enumerate(fig.axes):
        if i == 11:
            ax.axis('off')
            continue

        norms, idxs = zip(*sorted_norms_idxs[i])
        norms = np.array(norms)

        mu = np.mean(norms)
        sigma = np.std(norms)
        
        textstr = '\n'.join((
        r'Variance explained:',
        r'%.2f %%' % (pca_dct[i].explained_variance_ratio_.sum()*100),
        r'$\mu=%.2f$' % (mu, ),
        r'$\sigma=%.2f$' % (sigma, )
        ))


        ax.hist(norms, label = cdict[i], bins=50)
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.yaxis.set_tick_params(labelbottom=True)

        ax.set_title(f'Class {cdict[i]} (n = {len(idxs)})')
        ax.legend()
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            position=(0.6,0.5), bbox=props)

    #plt.suptitle(f'Distribution of norms of projections per class (Number of components: {num_components})', fontsize=15)
    fig.tight_layout()
    plt.savefig('norm_hist.png',facecolor='white', transparent=False)
    plt.show()

def vis_img(img, mean=None, std=None):
    if mean is not None:
        img = (img + (mean/std))/(1/std)  
    plt.imshow(img.permute(1, 2, 0).numpy())
    plt.show()

def find_highest_lowest(norm_dct):
    idx_dict = {}
    for label in norm_dct.keys():
        norms, idxs = zip(*norm_dct[label])
        threshold = np.percentile(np.array(norms),90)
        # Indexes of images above the 90 percentile:
        over_90 = np.array(norms)>=threshold
        idxs=np.array(idxs)
        idx_dict[label]=idxs[over_90]

    return idx_dict


def create_path_dct(real_dir, fake_dir):

    cropset = create_dataset(real_dir)

    img_dict = create_img_dict(cropset)
    norm_dct,pca_dct=create_norm_dict(cropset,img_dict,norm_dct_pickle=False, num_components=682)

    fake_set = create_dataset(fake_dir)

    fake_norm_dct = {}

    num_classes = len(fake_set.label_dict.keys())      

    reacgan_img_dct = create_img_dict(fake_set)

    for label in range(num_classes):
        imgs, idxs = zip(*reacgan_img_dct[label])
        stacked_imgs = torch.stack(imgs)
        pca = pca_dct[label]
        
        X_pca = pca.transform(stacked_imgs)
        X_proj = pca.inverse_transform(X_pca)

        reconstruct_diff = stacked_imgs - X_proj
        # Compute norms of error vectors:
        norm = np.linalg.norm(reconstruct_diff,axis=-1)
        # norm = compute_norm(reconstruct_diff)
        fake_norm_dct[label] = list(zip(norm,idxs))

    idx_over_90=find_highest_lowest(fake_norm_dct)
    path_dict = {i:[] for i in range(len(idx_over_90))}

    for i in idx_over_90:
        for idx in idx_over_90[i]:
            path_dict[i].append(fake_set.get_path(idx))

    return path_dict