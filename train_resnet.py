import torch
import torchvision.models as models
import torch.nn as nn

import torch
import numpy as np
from matplotlib import pyplot as plt

from time import time
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.optim import Adam
import logging

from cropsdata import create_dataset
from setup_aug_dir import add_fake_samples, add_pca_samples, add_pca_samples_over_under_90
from gen_pca import create_path_dct_small_set

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)

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

def training_loop(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, print_every, scheduler = None):
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

        save_checkpoint({
                'epoch': epoch ,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()
            })

    return model, train_losses, train_accs, val_losses, val_accs



def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


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

def testClassess(dataset,loader,model,device,mode='test'):
    if mode == 'test':
        mode_msg = 'Evaluating on test set' 
        print(mode_msg)
        logging.info(mode_msg)
    num_classes = len(dataset.label_dict.keys())
    classes = list(dataset.label_dict.keys())
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    with torch.no_grad():
        for x,y,_ in loader:
            inputs = x.to(device)
            labels = y.to(device)
            preds = model(inputs)
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

def train_resnet(img_dir, 
                fake_dir=None, 
                add_samples_per_class=None,
                pca_sampling=False,
                only_outliers=True,
                use_full_pca=False):

    # Naming files:
    if fake_dir is not None:
        model_name = fake_dir.name        
        fake_dir = fake_dir / 'imgs'
        
    else:
        model_name = img_dir.stem
    fake_dir_save = Path('/data/saved_classifier_logs')
    if not fake_dir_save.exists():
        fake_dir_save.mkdir()
    fake_dir_str = fake_dir_save.as_posix() + '/' + model_name
    hard_case_plot_name =fake_dir_str + f'_AugEval_hard_cases_{str(add_samples_per_class)}samples.png'
    prediction_plot_name = fake_dir_str + f'_AugEval_predictions_{str(add_samples_per_class)}samples.png'
    log_name = fake_dir_str + f'_AugEval_{str(add_samples_per_class)}samples.log'
    if pca_sampling:
        if only_outliers:
            hard_case_plot_name =fake_dir_str + f'_AugEval_hard_cases_{str(add_samples_per_class)}samples_PCA_only_outliers.png'
            prediction_plot_name = fake_dir_str + f'_AugEval_predictions_{str(add_samples_per_class)}samples_PCA_only_outliers.png'
            log_name = fake_dir_str + f'_AugEval_{str(add_samples_per_class)}samples_PCA_only_outliers.log'
        else:
            hard_case_plot_name =fake_dir_str + f'_AugEval_hard_cases_{str(add_samples_per_class)}samples_PCA_also_normal.png'
            prediction_plot_name = fake_dir_str + f'_AugEval_predictions_{str(add_samples_per_class)}samples_PCA_also_normal.png'
            log_name = fake_dir_str + f'_AugEval_{str(add_samples_per_class)}samples_PCA_also_normal.log'
        if use_full_pca:            
            if only_outliers:
                hard_case_plot_name =fake_dir_str + f'_AugEval_hard_cases_{str(add_samples_per_class)}samples_FULL_PCA_only_outliers.png'
                prediction_plot_name = fake_dir_str + f'_AugEval_predictions_{str(add_samples_per_class)}samples_FULL_PCA_only_outliers.png'
                log_name = fake_dir_str + f'_AugEval_{str(add_samples_per_class)}samples_FULL_PCA_only_outliers.log'
            else:
                hard_case_plot_name =fake_dir_str + f'_AugEval_hard_cases_{str(add_samples_per_class)}samples_FULL_PCA_also_normal.png'
                prediction_plot_name = fake_dir_str + f'_AugEval_predictions_{str(add_samples_per_class)}samples_FULL_PCA_also_normal.png'
                log_name = fake_dir_str + f'_AugEval_{str(add_samples_per_class)}samples_FULL_PCA_also_normal.log'


    if Path(log_name).exists():
        Path(log_name).unlink()
    logging.basicConfig(level=logging.INFO, filename=log_name,
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    if pca_sampling:
        title_msg = f'Starting {model_name}, {add_samples_per_class} samples, PCA.'
        if use_full_pca:
            title_msg = f'Starting {model_name}, {add_samples_per_class} samples, (full set) PCA.'
        if only_outliers:
            title_msg += 'Only outliers.'
        else:
            title_msg += 'Both outliers and normal data.'
    else:
        title_msg = f'Starting {model_name}, {add_samples_per_class} samples. Non PCA.' 
    logging.info(title_msg)
    logging.info('Creating dataset...')
    original_num_samples = len(list(img_dir.glob('*/*')))
    logging.info(f'Size of dataset: {original_num_samples}.')

    if pca_sampling:
        # If sampling using PCA, first find all files above or below the 90th percentile of
        # projection norms:
        logging.info('Adding fake samples via subspace sampling.')
        if not use_full_pca:
            path_dict_over90, path_dict_under90 = create_path_dct_small_set(img_dir,fake_dir)
        else:
            path_dict_over90, path_dict_under90 = create_path_dct_small_set(img_dir,fake_dir,use_full_pca=True)
        
        # If only sample over 90 percentile:
        if only_outliers:
            img_dir = add_pca_samples(img_dir, fake_dir, add_samples_per_class,path_dict_over90)
        # If sample from both > 90th percentile and from rest:
        else:
            img_dir = add_pca_samples_over_under_90(img_dir, fake_dir, add_samples_per_class,path_dict_over90, path_dict_under90)

        augmented_num_samples = len(list(img_dir.glob('*/*')))
        logging.info(f'Size of augmented dataset: {augmented_num_samples}.')
        pass


    elif fake_dir is not None:
        logging.info('Adding fake samples.')
        img_dir = add_fake_samples(img_dir, fake_dir, add_samples_per_class)

        augmented_num_samples = len(list(img_dir.glob('*/*')))
        logging.info(f'Size of augmented dataset: {augmented_num_samples}.')

    train_val_data = create_dataset(img_dir)

    BATCH_SIZE = 64
    logging.info('Batchsize: 64')
    logging.info('Splitting into training and validation set...')
    train_set, val_set= train_test_split(train_val_data, test_size=0.25,random_state=RANDOM_STATE)

    train_loader  = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader    = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    logging.info(f'There are {len(train_set)} train samples, {len(val_set)} validation samples.')
    logging.info(f'In total: {len(train_set)+len(val_set)}')

    logging.info('Starting training...')

    resnet = models.resnet18(pretrained=False)
    resnet.fc = nn.Linear(512, 11)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(resnet.parameters(), lr=5e-5, weight_decay=0.0001)
    num_epochs = 25
    print_every=len(train_loader)//2 +2
    _, train_losses, train_accs, val_losses, val_accs = training_loop(resnet, optimizer, loss_fn, train_loader, val_loader, num_epochs, print_every)

    logging.info('Done!')
    model_path = log_name[:-4] + 'classifier.pth'
    fpath = './resnet_real_classifier.pth'
    torch.save(resnet.state_dict(), model_path)
    logging.info(f'Model saved to {fpath}')


    test_msg = 'Evaluating on test set.'
    logging.info(test_msg)

    test_dir = Path('/data') / 'test_crops'
    test_data = create_dataset(test_dir)

    # Asserting that there is no data leakage:
    train_files = [f.name for f in img_dir.glob('*/*')]
    test_files = [f.name for f in test_dir.glob('*/*')]
    for f in train_files:
        if f in test_files:
            raise ValueError(f'File {f} in both train set and test set!')

    assert len(list(set(train_files) & set(test_files))) == 0, \
        f"Intersection of train and test files: {list(set(train_files) & set(test_files))}"


    test_loader  = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

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
    testClassess(test_data, test_loader, resnet, device,mode='test')

    logging.info('\nFinished!')

    print('Removing temporary dataset...')
    for subdir in img_dir.iterdir():
        for f in subdir.iterdir():
            if f.is_file():
                f.unlink()
        subdir.rmdir()
    img_dir.rmdir()