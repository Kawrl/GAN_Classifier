from GANtest import gantest
from pathlib import Path

for dir in ['/data/Baseline_DiffAug_200',
        '/data/ACGAN_200']:
    img_dir = Path(dir)
    gantest(img_dir)

