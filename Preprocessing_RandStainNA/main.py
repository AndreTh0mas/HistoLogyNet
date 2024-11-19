'''
Random Stain Normalization and Augmentation (RandStainNA) is a hybrid framework 
designed to fuse stain normalization and stain augmentation to generate more realistic stain variations. 
'''

import os
from randstainna import RandStainNA
import cv2



randstainna = RandStainNA(
    yaml_file = './preprocessing/output/random_images.yaml',
    std_hyper = 0.0,
    distribution = 'normal',
    probability = 1.0,
    is_train = False
)

dir_path = 'data/original/'
img_list = os.listdir(dir_path)


save_dir_path = 'data/augmented'
if not os.path.exists(save_dir_path):
    os.mkdir(save_dir_path)

for img_path in img_list:
    img = randstainna(cv2.imread(dir_path+img_path))
    save_img_path = save_dir_path + '/{}'.format(img_path.split('/')[-1])
    cv2.imwrite(save_img_path,img)


