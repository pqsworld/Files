from models.UnSuperPointNet_small_8_theta_student_block_v2_0814 import *
from pathlib import Path
from collections import deque, OrderedDict
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import copy
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

net = UnSuperPointNet_small_8_theta_student_block_v2()

print('# net parameters:', sum(param.numel() for param in net.parameters()))
print('# net parameters:', sum(param.numel() for param in net.parameters()) * 4 / 1024 , 'KB')