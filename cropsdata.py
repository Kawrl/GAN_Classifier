import torch

from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path


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

def create_dataset(img_dir):
    real_set_transform = transforms.Compose([
        TiffToTensor(),
        RectangularResize([64,32]),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x),
        CustomNormalization(new_min=0,new_max=1, source_min=0, source_max=65535),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    return CropsData(img_dir,real_set_transform)

