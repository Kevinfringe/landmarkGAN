import torch
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from PIL import Image

from generator import Pix2PixGenerator
from dataset import jaffeDataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_img_path = "./jaffedbase_official/test_data_jpg/"

checkpoint_path = "./100_first_train_checkpoint.pth"

output_exp_path = "./testing_output_exp/"
output_lm_path = "./testing_output_lm/"

exp_dict = {
    "AN": 0,
    "DI": 1,
    "FE": 2,
    "HA": 3,
    "NE": 4,
    "SA": 5,
    "SU": 6
}

def text_to_label(text):
    label = torch.zeros((7,))
    label[exp_dict[text]] = 1

    return label

transform = transforms.Compose([
        # resize image.
        transforms.Resize([256, 256]),
        # transform data into Tensor
        transforms.ToTensor()
    ])
transform_output = transforms.ToPILImage()
# define generators and load their weights.
checkpoints = torch.load(checkpoint_path)

G_lm = Pix2PixGenerator(label=torch.zeros((1, 7))).to(device)
G_exp = Pix2PixGenerator(in_channels=6, label=torch.zeros((1, 7))).to(device)

G_lm.load_state_dict(checkpoints['G_l_state_dict'])
G_exp.load_state_dict(checkpoints['G_e_state_dict'])



# to testify the model, we don't need dataset and dataloader.
for file in os.listdir(test_img_path):
    input_img = Image.open(os.path.join(test_img_path, file))
    input_img = transform(input_img).to(device)
    input_img = input_img.view(1, 3, 256, 256)

    # assign labels to generator.
    parts = file.split(".")
    exp_type = parts[1][:2]

    # try to generate all other expression type of images.
    for exp in exp_dict.keys():
        if exp_type != exp:
            # generate corresponding label.
            lm_gen_label = text_to_label(exp_type).to(device)
            exp_gen_label = text_to_label(exp).to(device)

            # assign label to generators
            G_lm.assign_label(lm_gen_label)
            G_exp.assign_label(exp_gen_label)

            # generate output image.
            lm_output = G_lm(input_img)
            exp_output = G_exp(torch.cat((input_img, lm_output), dim=1))

            lm_output = transform_output(lm_output)
            exp_output = transform_output(exp_output)

            # save output images.
            lm_img_name = f"{parts[0]}.{parts[1]}.{exp}.{parts[2]}.jpg"
            exp_img_name = f"{parts[0]}.{parts[1]}.{exp}.{parts[2]}.jpg"

            lm_output.save(os.path.join(output_lm_path, lm_img_name))
            exp_output.save(os.path.join(output_exp_path, exp_img_name))

