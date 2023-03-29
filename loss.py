import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

from dataset import jaffeDataset

# posed_img_path = "./pose_set"
# genuine_img_path = "./genuine_set"
# lm_posed_path = "./lm_posed"
# lm_genuine_path = "./lm_genuine"
# lm_img_posed_path = "./lm_image_posed"
# lm_img_genuine_path = "./lm_image_genuine"

class GANLoss(nn.Module):
    """
        Define the GANLoss class
        Note that only BCE loss is consider in current situation.
        Code is inspired by original Pix2Pix code:
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        # keep below two variables away from gradient calculation.
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, prediction, target_is_real):

        """
            Create label tensor with the size shape of input.
        :param prediction: prediction output from the discriminator
        :param target_is_real: bool var to indicate whether the GT label is for real images(samples)
        or fake images(output from discriminator)
        :return: A label tensor fill with GT label, and with the same size as input.
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return self.loss(prediction, target_tensor)


class VGGPerceptualLoss(nn.Module):
    """
        The Perceptual loss is computed by deploying a pre-trained VGG feature extractor,
        and then compute the L1 distance between V(x) and V(y).
        This loss function is for generator.
        Code is inpired by: https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
    """
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        # Only make use of specific layers of VGG, according to paper
        # Johnson et al. 2016 "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        # Freeze all the parameters in selected layers.
        for block in blocks:
            for param in block.parameters():
                param.requires_grad = False

        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        # The mean and std for ImageNet, which VGG is pre-trained on.
        # Keep them constant.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=None, style_layers=[]):
        # Normalize the input and target to [0, 1].
        if feature_layers is None:
            feature_layers = [0, 1, 2, 3]
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
        target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)

            # Style reconstruction loss mentioned in original paper.
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


class IdentityLoss(nn.Module):
    """
        The identity loss is added to preserve the identity information of
        output image with respect to input image.
        Here a pretrained  face recognition model -- FaceNet is deployed here.
        Only the output embedding (512 dimensional) is used, and image with the same
        identity, the embedding should be very similar.
        --generated image : output from generator, (N * 3 * 256 *256)
        --target-image: target image in dataset
    """
    def __init__(self):
        super(IdentityLoss, self).__init__()
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        self.loss = nn.L1Loss()

    def __call__(self, generated_image, target_image):
        # First both image need to be resized to (160 * 160)
        transform = transforms.Resize([160, 160])
        generated_image = transform(generated_image)
        target_image = transform(target_image)
        embedding_gen = self.model(generated_image)
        embedding_target = self.model(target_image)

        return self.loss(embedding_gen, embedding_target)


class LandmarkLoss(nn.Module):
    """
        Landmark Loss is to calculate the L1 loss between generated image (landmark region) and
        target_image (landmark region).
    """
    def __init__(self, generated_img=None, target_img=None,
                 lm_array=None, original_size=None):
        super(LandmarkLoss, self).__init__()
        self.gen_img = generated_img
        self.tar_img = target_img
        self.lm_array = lm_array
        self.original_size = original_size
        self.loss = nn.L1Loss(reduction='mean')


    def assign_fields(self, generated_img, target_img,
                 lm_array, original_size):
        self.gen_img = generated_img
        self.tar_img = target_img
        self.lm_array = lm_array
        self.original_size = original_size


    def get_lm_patches(self, input):
        """
            This function is to extract the 3*3 region around landmarks
            and form them in (N * 68 * 3 * 3) tensor, since there are 68 landmarks
            detected.
        :return: (N * 68 * 3 * 3) tensor
        """
        # First since the target img has been resized, so the
        # landmark coordinates also need resize.
        # for i in range(self.lm_array.size(0)):
        #     self.lm_array[i] = self.lm_array[i] * 256 / self.original_size[i]

        batch_size = input.size(0)
        # print("Batch size is :" + str(batch_size))
        # print("lm_array size is :" + str(self.lm_array.size()))
        res = torch.ones_like(input) * 255
        lm_length = (self.lm_array.size(2) - 2) // 2

        # Assign pixel values to the landmark regions.
        for i in range(batch_size):

            for j in range(lm_length):
                x_cor = int(self.lm_array[i, 0, 2 + j]) if (self.lm_array[i, 0, 2 + j] <= 254) else 254
                x_cor = x_cor if x_cor >= 1 else 1
                y_cor = int(self.lm_array[i, 0, 2 + lm_length + j]) if self.lm_array[i, 0, 2 + lm_length + j] <= 254 else 254
                y_cor = y_cor if y_cor >= 1 else 1

                res[i, :, x_cor - 1: x_cor + 1, y_cor - 1: y_cor + 1] = input[i, :, x_cor - 1: x_cor + 1, y_cor - 1:y_cor + 1]

        # print(torch.count_nonzero(res))
        return res


    def __call__(self):
        gen_img_lm = self.get_lm_patches(self.gen_img)
        tar_img_lm = self.get_lm_patches(self.tar_img)

        return self.loss(gen_img_lm, tar_img_lm)






# # Test for faceReconstruction loss
# # Transformation on data.
# transform = transforms.Compose([
#     # resize image.
#     transforms.Resize([256, 256]),
#     # transform data into Tensor
#     transforms.ToTensor(),
#     # Normalize data into range(-1, 1)
#     # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
# ])
# train_set = CustomDataset(lm_img_path=lm_img_posed_path, input_img_path=posed_img_path, target_img_path=genuine_img_path,
#                                       target_lm_path=lm_genuine_path, input_transform=transform, target_img_transform=transform)
# trainloader = DataLoader(train_set, batch_size=1, shuffle=True)
# lm_loss = FaceReconstructionLoss()
# for batch_idx, (input_lm_img, input_img, PG_label, target_img, lm_array, original_size) in enumerate(trainloader):
#     # input_img = torch.randn((64, 3, 256, 256)) * 255
#     lm_loss.assign_fields(input_img, target_img, lm_array, original_size)
#     loss = lm_loss()
#     print(loss.item())
#     break

# test for landmark loss
# lm_loss = LandmarkLoss()
# generated_img = torch.rand((1,3,256,256))
# target_img = torch.rand((1,3,256,256))
# df = pd.read_csv("./jaffedbase_official/lm_csv/KA.AN1.39.csv")
# lm_csv = df.to_numpy()
# lm_csv = torch.tensor(lm_csv).view(1,1,138)
#
# lm_loss.assign_fields(generated_img,target_img,lm_csv,256)
# loss = lm_loss()
# print(loss.item())
