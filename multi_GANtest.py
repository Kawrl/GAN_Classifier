from GANtest import gantest
from pathlib import Path

for dir in ['/data/ACGAN_1000',
            '/data/Baseline_DiffAug_1000']:
    img_dir = Path(dir)
    gantest(img_dir)

