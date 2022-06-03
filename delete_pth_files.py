from pathlib import Path
from argparse import ArgumentParser

def dir_path(path):
    pathlib_path = Path(path)
    if pathlib_path.is_dir():
        return pathlib_path
    else:
        raise f"readable_dir:{path} is not a valid path"

def delete_pth(dir):
    for file in dir.glob('**/*.pth'):
        print(f'File {file.name} is now deleted.')
        file.unlink()

def load_parser():
    parser = ArgumentParser(add_help=True)
    parser.add_argument("--dir", type=dir_path, default="./")
    args = parser.parse_args()    

    return args

if __name__ == '__main__':
    args = load_parser()
    img_dir= args.dir

    delete_pth(img_dir)

