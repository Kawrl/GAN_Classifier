from GANtest import gantest
from pathlib import Path

for dir in ['/data/ACGAN_200']:
    img_dir = Path(dir)
    gantest(img_dir)

