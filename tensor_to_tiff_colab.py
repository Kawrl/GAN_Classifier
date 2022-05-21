import torch
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import json
from pathlib import Path
import torchvision.transforms.functional as F
from argparse import ArgumentParser



def generated_to_tiff(image_tensor: torch.tensor, file_name: str):
    # Rescale to unsigned 16-bit: [0,65535]
    source_min = -1
    source_max = 1
    new_min = 0
    new_max = 65535
    rescaled_img = (image_tensor-source_min)*(new_max-new_min)/(source_max-source_min) + new_min
    
    # Convert to numpy and recast to 16 bit integer.
    # Only the first channel, since target image is one channel.
    rescaled_img = rescaled_img[0].numpy().astype(np.int16)
    
    # Convert to PIL image and save.
    tiff_img = F.to_pil_image(rescaled_img, mode='I;16')
    tiff_img.save(file_name,'TIFF')


def dir_path(path):
    pathlib_path = Path(path)
    if pathlib_path.is_dir():
        return pathlib_path
    else:
        #raise f"readable_dir:{path} is not a valid path"
        path.mkdir()


def load_parser():
    parser = ArgumentParser(add_help=True)
    #parser.add_argument("--save_dir", type=dir_path, default="./")
    parser.add_argument("--tensor_dir", type=dir_path, default="./")
    args = parser.parse_args()    

    return args

if __name__ == '__main__':
    args = load_parser()
    #save_dir = args.save_dir
    tensor_dir = args.tensor_dir
    fake_dir = tensor_dir / 'fake'
    save_dir = tensor_dir / 'imgs'
    print('Saving images at:', save_dir)    

    # Loading the tensors and concatenating them to three single tensors 
    img_list, lbl_list, z_list = [], [], []

    for file in sorted(list(fake_dir.glob('*.pt'))):
        tensor = torch.load(file)
        if file.stem.startswith('all_labels'):
            lbl_list.append(tensor)
        elif file.stem.startswith('all_images'):
            img_list.append(tensor)
        elif file.stem.startswith('all_zs'):
            z_list.append(tensor)

    print('Done saving all images.')
    if not save_dir.exists():
        save_dir.mkdir()
            

    label_tensor=torch.cat(lbl_list)
    img_tensor=torch.cat(img_list)
    z_tensor=torch.cat(z_list)

    json_dict = {'generated_images':[]}
    for i in range(11):
        fpath = save_dir / str(i)
        if not fpath.exists():
            fpath.mkdir(parents=True)

    for idx, (img, label, zs) in enumerate(zip(img_tensor, label_tensor, z_tensor)):
        f_end = str(idx)+'.tiff'
        fpath = save_dir / str(label.item()) / f_end
        generated_to_tiff(img, fpath)

        fname = str(idx)
        json_entry = {'file_name':fname, 'label':label.item(), 'z_vector':zs.tolist()}
        json_dict['generated_images'].append(json_entry)

    outfile_name = save_dir / 'img_json.json'
    json_string = json.dumps(json_dict)
    with open(outfile_name, 'w', encoding='utf8') as outfile:
        json.dump(json_dict, outfile, sort_keys = True, indent = 4,
                ensure_ascii = False)