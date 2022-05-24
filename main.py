import os
import numpy as np
import pandas as pd
from tqdm import tqdm

meta_file = './waterbirds/metadata.csv'
df = pd.read_csv(
    meta_file,
    keep_default_na=False)

dirs = np.unique([img_filename.split('/')[0]
                 for img_filename in df['img_filename']])

# loop dirs, list all files inside each dir
for dir in tqdm(dirs):
    dir = './waterbirds/images/' + dir
    print(dir)
    # loop each file in dir
    for file in tqdm(os.listdir(dir)):
        # if file ends with .jpg, and not ends with .detect.jpg
        if file.endswith('.jpg') and not file.endswith('.detect.jpg'):
            os.system(
                'python tools/demo/demo_image.py --config_file sgg_configs/vgattr/vinvl_x152c4.yaml --img_file ' + dir + '/' + file + ' MODEL.WEIGHT pretrained_model/vinvl_vg_x152c4.pth MODEL.ROI_HEADS.NMS_FILTER 1 MODEL.ROI_HEADS.SCORE_THRESH 0.2 TEST.IGNORE_BOX_REGRESSION False')
