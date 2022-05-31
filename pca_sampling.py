from argparse import ArgumentParser
from train_resnet import train_resnet
from pathlib import Path

def dir_path(path):
    pathlib_path = Path(path)
    if pathlib_path.is_dir():
        return pathlib_path
    else:
        raise f"readable_dir:{path} is not a valid path"

def load_parser():
    parser = ArgumentParser(add_help=True)
    parser.add_argument("--img_dir", type=dir_path, default="./")
    parser.add_argument("--fake_dir", type=dir_path, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--only_outliers", action="store_true")
    parser.add_argument("--full_pca", action="store_true")
    args = parser.parse_args()    

    return args

if __name__ == '__main__':
    args = load_parser()
    img_dir= args.img_dir
    fake_dir=args.fake_dir
    num_samples=args.num_samples
    only_outliers = args.only_outliers
    full_pca = args.full_pca
    print('Only outliers?', only_outliers)

    train_resnet(img_dir, fake_dir, num_samples, pca_sampling=True,only_outliers=only_outliers,use_full_pca=full_pca)