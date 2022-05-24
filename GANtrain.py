import torch
import torchvision.models as models
import torch.nn as nn

import os
import random

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from PIL import ImageOps, Image
import torch
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

from time import time
from torch.utils.data import Dataset, DataLoader, random_split
from itertools import chain
from PIL import Image
from pathlib import Path

from sklearn.model_selection import train_test_split
from torch.optim import Adam

import logging

from argparse import ArgumentParser

class RectangularResize(object):
    """
    Rectangular center crop of image.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size.
    """

    # As per https://pytorch.org/tutorials/beginner/data_loading_tutorial.html:
    def __init__(self, output_size):
        assert isinstance(output_size, list), f"Wrong output type. Expected list, got {type(output_size)}"
        self.output_size = output_size

    def __call__(self, img):
        return transforms.functional.resize(img,self.output_size)

    def __repr__(self):
        return self.__class__.__name__

class TiffToTensor(object):

    def __call__(self, img):
        # Assuming the image is a PIL image.
        # Pixel values are stored in a list and then converted
        # to torch tensor. Should not change any values.
        # Also reshapes it to its original shape (c x h x w)
        return torch.tensor(list(img.getdata())).reshape(1,img.size[1],img.size[0])

    def __repr__(self):
        return self.__class__.__name__

class CustomNormalization(object):
    def __init__(self, new_min=0,new_max=1, source_min=None, source_max=None):
        self.new_min = new_min
        self.new_max = new_max
        self.source_min = source_min
        self.source_max = source_max

    def normalize_tensor(self,img):
        if self.source_min == None or self.source_max == None:
            self.source_min, self.source_max = img.min(), img.max()
        normalized_tensor = (img-self.source_min)*(self.new_max-self.new_min)/(self.source_max-self.source_min) + self.new_min
        return normalized_tensor 

    def __call__(self, img):
        return self.normalize_tensor(img)

    def __repr__(self):
        return self.__class__.__name__

class CropsData(Dataset):
    
    def __init__(self, root, transform):
        """Constructor
        
        Args:
            root (Path/str): Filepath to the data root, e.g. './small_train'
            transform (Compose): A composition of image transforms, see below.
        """

        root = Path(root)
        if not (root.exists() and root.is_dir()):
            raise ValueError(f"Data root '{root}' is invalid")
            
        self.root = root
        self.transform = transform

        self.label_dict = self.create_label_dict()
        self.labels = list(self.label_dict.values())
        
        # Collect samples, both cat and dog and store pairs of (filepath, label) in a simple list.
        self._samples = self._collect_samples()

    def create_label_dict(self):
        label_dict = {str(i):i for i in range(11)}
        return label_dict

    def get_path(self, index):

        path, label = self._samples[index]
        return path
            
    def __getitem__(self, index):
        """Get sample by index
        
        Args:
            index (int)
        
        Returns:
             The index'th sample (Tensor, int)
        """
        # Access the stored path and label for the correct index
        path, label = self._samples[index]
        # Load the image into memory
        with open(path, "rb") as f:
            img = Image.open(f)
            img.load()
        # Perform transforms, if any.
        if self.transform is not None:
            img = self.transform(img)
        return img, label, index
    
    def __len__(self):
        """Total number of samples"""
        return len(self._samples)
    
    def _collect_samples(self):
        """Collect all paths and labels
        
        Helper method for the constructor
        """
        all_paths_labels = []
        for sub_dir in self.root.iterdir():
            if not sub_dir.is_dir():
                continue
            paths = self._collect_imgs_sub_dir(sub_dir)
            if sub_dir.stem == 'W':
                label = self.label_dict['10']
            else:
                label = self.label_dict[sub_dir.stem]
            
            path_and_label = [(path, label) for path in paths]
            all_paths_labels.extend(path_and_label)
            
        # Sorting is not strictly necessary, but filesystem globbing (wildcard search) is not deterministic,
        # and consistency is nice when debugging.
        return sorted(all_paths_labels, key=lambda x: x[0].stem)
     
    @staticmethod
    def _collect_imgs_sub_dir(sub_dir: Path):
        """Collect image paths in a directory
        
        Helper method for the constructor
        """
        if not sub_dir.exists():
            raise ValueError(f"Data root '{self.root}' must contain sub dir '{sub_dir.name}'")
        return sub_dir.glob("*.tiff")
    
    def get_sample_by_id(self, id_):
        """Get sample by image id
        
        Convenience method for exploration.
        The indices does not correspond to the image id's in the filenames.
        Here is a (rather inefficient) way of inspecting a specific image.
        
        Args:
            id_ (str): Image id, e.g. `dog.321`
        """
        id_index = [path.stem for (path, _) in self._samples].index(id_)
        return self[id_index]

def training_loop(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, print_every, scheduler = None):
    logging.info('Starting training.')
    device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
    model.to(device)
    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    for epoch in range(1, num_epochs+1):
        start_time = time()
        model, train_loss, train_acc = train_epoch(model,
                                                   optimizer,
                                                   scheduler,
                                                   loss_fn,
                                                   train_loader,
                                                   val_loader,
                                                   device,
                                                   print_every)
        val_loss, val_acc = validate(model, loss_fn, val_loader, device)
        epoch_message = (f"Epoch {epoch}/{num_epochs}: "
              f"Train loss: {sum(train_loss)/len(train_loss):.3f}, "
              f"Train acc.: {sum(train_acc)/len(train_acc):.3f}, "
              f"Val. loss: {val_loss:.3f}, "
              f"Val. acc.: {val_acc:.3f},"
              f"Time: {time()-start_time:.3f}")

        print(epoch_message)
        logging.info(epoch_message)
        train_losses.extend(train_loss)
        train_accs.extend(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    return model, train_losses, train_accs, val_losses, val_accs

def train_epoch(model, optimizer, scheduler, loss_fn, train_loader, val_loader, device, print_every):
    # Train:
    model.train()
    train_loss_batches, train_acc_batches = [], []
    num_batches = len(train_loader)
    for batch_index, (x, y, idx_) in enumerate(train_loader, 1):
        inputs = x.to(device)
        labels = y.to(device)
        optimizer.zero_grad()
        preds = model.forward(inputs)        
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        train_loss_batches.append(loss.item())

        hard_preds = preds.argmax(dim=1)
        acc_batch_avg = (hard_preds == labels).float().mean().item()
        train_acc_batches.append(acc_batch_avg)

        # If you want to print your progress more often than every epoch you can
        # set `print_every` to the number of batches you want between every status update.
        # Note that the print out will trigger a full validation on the full val. set => slows down training
        if print_every is not None and batch_index % print_every == 0:
            val_loss, val_acc = validate(model, loss_fn, val_loader, device)
            model.train()
            eval_message = (f"\tBatch {batch_index}/{num_batches}: "
                  f"\tTrain loss: {sum(train_loss_batches[-print_every:])/print_every:.3f}, "
                  f"\tTrain acc.: {sum(train_acc_batches[-print_every:])/print_every:.3f}, "
                  f"\tVal. loss: {val_loss:.3f}, "
                  f"\tVal. acc.: {val_acc:.3f}")
            print(eval_message)
            logging.info(eval_message)

    return model, train_loss_batches, train_acc_batches

def validate(model, loss_fn, val_loader, device):
    val_loss_cum = 0
    val_acc_cum = 0
    model.eval()
    with torch.no_grad():
        for batch_index, (x, y, idx_) in enumerate(val_loader, 1):
            inputs, labels = x.to(device), y.to(device)
            preds = model.forward(inputs)
            batch_loss = loss_fn(preds, labels)
            val_loss_cum += batch_loss.item()
            hard_preds = preds.argmax(dim=1)
            acc_batch_avg = (hard_preds == labels).float().mean().item()
            val_acc_cum += acc_batch_avg
    return val_loss_cum/len(val_loader), val_acc_cum/len(val_loader)


def plot_samples(dataset, mean, std, suptitle,fname):
    random_indexes = torch.randint(0,len(dataset),(25,))
    fig, axs = plt.subplots(5,5,figsize=(16,16))

    axs = axs.ravel()

    for idx,img_indx in enumerate(random_indexes):
        img, label, _ = dataset[img_indx.item()]
        img = (img + (mean/std))/(1/std)    
        image_data = img.permute(1, 2, 0).numpy()
        axs[idx].imshow(image_data)
        axs[idx].axis('off')
        axs[idx].set_title(f'Label: {label}', fontsize=15)
    plt.suptitle(suptitle, fontsize=30)
    plt.savefig(fname)

def testClassess(dataset,mode='test'):
    if mode == 'val':
        mode_msg = 'Evaluating on validation set' 
        print(mode_msg)
        logging.info(mode_msg)
        loader = val_loader
    if mode == 'test':
        mode_msg = 'Evaluating on test set' 
        print(mode_msg)
        logging.info(mode_msg)
        loader = test_loader
    num_classes = len(dataset.label_dict.keys())
    classes = list(dataset.label_dict.keys())
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    with torch.no_grad():
        for x,y,_ in loader:
            inputs = x.to(device)
            labels = y.to(device)
            preds = resnet(inputs)
            hard_preds = preds.argmax(dim=1)
            c = (hard_preds == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(num_classes):
        acc_info = 'Accuracy of class %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i])
        print(acc_info)
        logging.info(acc_info)

def dir_path(path):
    pathlib_path = Path(path)
    if pathlib_path.is_dir():
        return pathlib_path
    else:
        raise f"readable_dir:{path} is not a valid path"


def load_parser():
    parser = ArgumentParser(add_help=True)
    parser.add_argument("--model_dir", type=dir_path, default="./")
    args = parser.parse_args()    

    return args

if __name__ == '__main__':

    args = load_parser()
    model_dir = args.model_dir
    img_dir = model_dir / 'imgs'
    model_name = model_dir.name
    img_dir_str = model_dir.as_posix() + '/' + model_name

    log_name = model_name + '_GANtrain_log.log'
    real_plot_name = img_dir_str + '_GANtrain_realplt.png'
    hard_case_plot_name =img_dir_str + '_GANtrain_hard_cases.png'
    prediction_plot_name = img_dir_str + '_GANtrain_predictions.png'

    if Path(log_name).exists():
        Path(log_name).unlink()

    RANDOM_STATE = 42
    torch.manual_seed(0)
    #summary(resnet, (3, 64, 32))

    logging.basicConfig(filename=log_name, level=logging.INFO)

    real_set_transform = transforms.Compose([
        TiffToTensor(),
        RectangularResize([64,32]),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x),
        CustomNormalization(new_min=0,new_max=1, source_min=0, source_max=65535),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    train_val_path = img_dir
    test_path = Path.cwd() / 'test_crops'
    train_val_data = CropsData(train_val_path,real_set_transform)
    test_data = CropsData(test_path,real_set_transform)

    data_info = f'There are {len(train_val_data)} samples in the train/val set and {len(test_data)} in test set.'
    print(data_info)
    logging.info(data_info)

    plot_samples(dataset=train_val_data,
                mean=0.5,
                std=0.5,
                suptitle='Real set samples',
                fname=real_plot_name)

    BATCH_SIZE = 64

    train_set, val_set= train_test_split(train_val_data, test_size=0.25,random_state=RANDOM_STATE)
    train_loader  = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader    = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader   = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    loader_msg = f'There are {len(train_set)} train samples, {len(val_set)} validation samples, {len(test_data)} test samples. In total {len(train_set)+len(val_set)+len(test_data)}'
    print(loader_msg)
    logging.info(loader_msg)



    resnet = models.resnet18(pretrained=False)
    resnet.fc = nn.Linear(512, 11)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(resnet.parameters(), lr=1e-6, weight_decay=0.0001)
    num_epochs = 5
    print_every=len(train_loader)//2 +2
    _, train_losses, train_accs, val_losses, val_accs = training_loop(resnet, optimizer, loss_fn, train_loader, val_loader, num_epochs, print_every)

    test_loss_cum = 0
    test_acc_cum = 0
    resnet.eval()
    device = torch.device("cuda" if torch.cuda.is_available() 
                                    else "cpu")
    gtruth = []
    pred_list = []
    ixs_lst = []

    with torch.no_grad():
        for batch_index, (x, y, idx_) in enumerate(test_loader, 1):
            inputs = x.to(device)
            labels = y.to(device)
            preds = resnet.forward(inputs)
            batch_loss = loss_fn(preds, labels)
            test_loss_cum += batch_loss.item()
            hard_preds = preds.argmax(dim=1)
            ixs = np.where((hard_preds!=labels).cpu())
            ixs_lst.extend(idx_[ixs].tolist())
            acc_batch_avg = (hard_preds == labels).float().mean().item()
            gtruth.extend(labels.cpu().tolist())
            pred_list.extend(hard_preds.cpu().tolist())
            test_acc_cum += acc_batch_avg
    test_loss = test_loss_cum/len(test_loader), 
    test_accuracy=test_acc_cum/len(test_loader)
    test_msg = f'Accuracy on the test set is {100*test_accuracy:.3f}%.'
    logging.info(test_msg)
    print(test_msg)

    hard_cases = [test_data[ix] for ix in ixs_lst]
    hard_loader = DataLoader(hard_cases, 8)

    plot_samples(dataset=hard_cases,
                mean=0.5,
                std=0.5,
                suptitle='Hard cases',
                fname=hard_case_plot_name)

    hard_loader = DataLoader(hard_cases,64)

    resnet.eval()
    device = torch.device("cuda" if torch.cuda.is_available() 
                                    else "cpu")
    gtruth_small=[]
    pred_list_small=[]
    with torch.no_grad():
        for batch_index, (x, y, idx_) in enumerate(hard_loader, 1):
            inputs = x.to(device)
            labels = y.to(device)
            preds = resnet.forward(inputs)
            batch_loss = loss_fn(preds, labels)
            test_loss_cum += batch_loss.item()
            hard_preds = preds.argmax(dim=1)
            acc_batch_avg = (hard_preds == labels).float().mean().item()
            gtruth_small.extend(labels.cpu().tolist())
            pred_list_small.extend(hard_preds.cpu().tolist())
            break

    fig, axs = plt.subplots(5,5,figsize=(16,16))
    axs = axs.ravel()

    cases = zip(inputs[:25],labels[:25],hard_preds[:25])
    mean=0.5
    std=0.5
    for idx,(img, label,prediction) in enumerate(cases):
        #img = samples[0]
        #prediction = samples[1]
        img = (img + (mean/std))/(1/std)    
        image_data = img.cpu().permute(1, 2, 0).numpy()
        axs[idx].imshow(image_data)
        axs[idx].axis('off')
        axs[idx].set_title(f'Prediction: {prediction}. \nGround truth: {label}', fontsize=10)
    plt.suptitle('Predictions and true labels', fontsize=30)
    plt.savefig(prediction_plot_name)

    logging.info('---'*10)
    testClassess(train_val_data, mode='test')

    logging.info('\nFinished!')