import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# define the dataset variables.
train_set_path = "./jaffedbase_official/jaffedbase"
lm_img_path = "./jaffedbase_official"
test_set_path = "./jaffedbase_official/jaffedbase/test_data"

class JAFFEDataset(Dataset):
    def __init__(self, train_set_path, ):

