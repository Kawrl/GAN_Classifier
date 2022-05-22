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
    args = parser.parse_args()    

    return args

if __name__ == '__main__':
    args = load_parser()
    img_dir= args.img_dir
    fname = 'files_' + img_dir.name + '.txt'

    training_files200 = [f.name for f in list(img_dir.glob('*/*.tiff'))]
    with open(fname,'w') as f:
        for file in training_files200:
            _ = file + '\n'
            f.write(_)
