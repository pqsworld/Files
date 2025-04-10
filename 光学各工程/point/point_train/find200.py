import os
path="/ssd/share/op_train_data/desc_op/descop_200_180/DeepDesc_Part1"
from PIL import Image
import numpy as np
for root,_,files in os.walk(path):
    for file in files:
        if ".bmp" in file:
            pathimg=os.path.join(root,file)
            img=Image.open(pathimg).convert("L")
            img=np.array(img)
            if img.shape[0]==200:
                print(pathimg)
        
