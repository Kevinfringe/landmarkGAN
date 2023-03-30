import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import dataset
import generator
import discriminator
import loss

# define training description.
training_description = "with_originalimg_id_5_facere_100_others_10_rm_normal_target"

# sys.stdout = open("log_01.txt", "w")
train_set_input_path = "./jaffedbase_official/train_set_final/"
train_set_target_path = "./jaffedbase_official/train_set_jpg/"
lm_image_path = "./jaffedbase_official/lm_img/"
lm_csv_path = "./jaffedbase_official/lm_csv/"


def train(args, G_l, D_l, G_e, D_e, device, trainloader, optimizer_G_l, optimizer_D_l, optimizer_G_e, optimizer_D_e, epoch):
    print("********* Train Epoch " + str(epoch) + " start here *********")
    start = time.time()
    G_l.train()
    D_l.train()
    G_e.train()
    D_e.train()

    D_l_loss = 0.0
    G_l_loss = 0.0
    D_e_loss = 0.0
    G_e_loss = 0.0

    gan_loss = loss.GANLoss().to(device)
    lm_loss = loss.LandmarkLoss().to(device)
    id_loss = loss.IdentityLoss().to(device)
    # re_loss = loss.FaceReconstructionLoss().to(device)
    re_loss = nn.HuberLoss().to(device)

    for batch_idx, (img, lm_img, target_img, lm_array, lm_gen_label, exp_gen_label) in enumerate(trainloader):
        img, lm_img, target_img, lm_array, lm_gen_label, exp_gen_label = img.to(device), lm_img.to(device), target_img.to(device), \
                                                                         lm_array.to(device), lm_gen_label.to(device), exp_gen_label.to(device)

        # Train G_l first.
        # pass in the label.
        G_l.assign_label(lm_gen_label)
        # First train discriminator.
        lm_img_fake = G_l(img)
        D_l_real = D_l(img, lm_img)
        D_real_l_loss = gan_loss(D_l_real, target_is_real=True)
        D_l_fake = D_l(img, lm_img_fake.detach())
        D_l_fake_loss = gan_loss(D_l_fake, target_is_real=False)
        D_l_train_loss = (D_real_l_loss + D_l_fake_loss) / 2

        optimizer_D_l.zero_grad()
        D_l_train_loss.backward()
        optimizer_D_l.step()

        # Then train generator.
        # Gan loss for generator.
        D_l_pred = D_l(img, lm_img_fake)
        G_l_gan_loss = gan_loss(D_l_pred, target_is_real=True)
        # landmark loss
        lm_loss.assign_fields(lm_img_fake, lm_img, lm_array, 256)
        G_l_lm_loss = lm_loss()

        G_l_loss = args.lambda_lm * G_l_lm_loss + G_l_gan_loss

        optimizer_G_l.zero_grad()
        G_l_loss.backward()
        optimizer_G_l.step()

        # Then train G_e
        # First train D_e
        # pass in the label.
        G_e.assign_label(exp_gen_label)
        # First train discriminator.
        exp_img_fake = G_e(torch.cat((img, lm_img_fake.detach()), dim=1))
        D_e_real = D_e(img, target_img)
        D_real_e_loss = gan_loss(D_e_real, target_is_real=True)
        D_e_fake = D_e(img, exp_img_fake.detach())
        D_e_fake_loss = gan_loss(D_e_fake, target_is_real=False)
        D_e_train_loss = (D_real_e_loss + D_e_fake_loss) / 2

        optimizer_D_e.zero_grad()
        D_e_train_loss.backward()
        optimizer_D_e.step()

        # Then train generator.
        # Gan loss for generator.
        D_e_pred = D_e(img, exp_img_fake)
        G_e_gan_loss = gan_loss(D_e_pred, target_is_real=True)

        G_e_id_loss = id_loss(exp_img_fake, target_img)
        G_e_face_re_loss = re_loss(exp_img_fake, target_img)
        G_train_e_loss = G_e_gan_loss + args.lambda_id * G_e_id_loss + args.lambda_re * G_e_face_re_loss

        optimizer_G_e.zero_grad()
        G_train_e_loss.backward()
        optimizer_G_e.step()

        D_l_loss += D_l_train_loss.item()
        G_l_loss += G_l_loss.item()
        D_e_loss += D_e_train_loss.item()
        G_e_loss += G_train_e_loss.item()

        # Print the log
        if batch_idx % args.log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tD_l_loss: {:.6f}\tG_l_loss: {:.6f}\n D_e_loss: {:.6f} G_e_loss: {:.6f}'
                ' G_e_id_loss: {:.6f} G_re_loss: {:.6f} G_l_lm_loss: {:.6f}'.format(
                    epoch, batch_idx * len(img), len(trainloader.dataset),
                           100. * batch_idx / len(trainloader), D_l_train_loss.item(), G_l_loss.item(),
                    D_e_train_loss.item(), G_train_e_loss.item(),
                    G_e_id_loss.item(), G_e_face_re_loss.item(), G_l_lm_loss.item()))

    D_l_loss /= len(trainloader.dataset)
    G_l_loss /= len(trainloader.dataset)
    D_e_loss /= len(trainloader.dataset)
    G_e_loss /= len(trainloader.dataset)

    end = time.time()
    print('\nTrain Epoch {} finished\nAverage Discriminator_l loss : {:.6f}, G_l loss: {:.6f}, '
          'Discriminator_e loss : {:.6f}, G_e loss: {:.6f} time: {:.6f}'.format(
        epoch, D_l_loss, G_l_loss, D_e_loss, G_e_loss, end - start
    ))

    # save model
    if epoch % 100 == 0:
        checkpoint_path = './{}_{}_checkpoint.pth'.format(epoch, training_description)
        torch.save({
            'G_l_state_dict': G_l.state_dict(),
            'optimizer_G_l_state_dict': optimizer_G_l.state_dict(),
            'D_l_state_dict': D_l.state_dict(),
            'optimizer_D_l_state_dict': optimizer_D_l.state_dict(),
            'G_e_state_dict': G_e.state_dict(),
            'optimizer_G_e_state_dict': optimizer_G_e.state_dict(),
            'D_e_state_dict': D_e.state_dict(),
            'optimizer_D_e_state_dict': optimizer_D_e.state_dict()
        }, checkpoint_path)

    # sys.stdout.flush()


def main():
    start = time.time()
    # Training settings
    parser = argparse.ArgumentParser(description="LandmarkGAN")
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training. default=1')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing. default = 1000')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train. default = 200')
    parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                        help='learning rate')
    parser.add_argument('--betas', default=(0.5, 0.999),
                        help='betas in Adam optimizer.')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--gamma', type=float, default=0.5, metavar='M',
                        help='Learning rate step gamma (default: 0.5)')
    parser.add_argument('--lambda_lm', type=int, default=2, metavar='M',
                        help='weight parameter for landmark loss in generator')
    parser.add_argument('--lambda_id', type=int, default=1, metavar='M',
                        help='weight parameter for identity loss in generator')
    parser.add_argument('--lambda_re', type=int, default=100, metavar='M',
                        help='weight parameter for reconstruction loss in generator')
    parser.add_argument('--lambda_gan', type=int, default=10, metavar='M',
                        help='weight parameter for gan loss in generator')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    print("cuda is available on this machine :" + str(use_cuda))

    device = torch.device("cuda" if use_cuda else "cpu")

    #### First train from posed image to genuine image.

    # For both target and input image, only normalize them into [-1, 1]
    # Transformation on data.
    transform = transforms.Compose([
        # resize image.
        transforms.Resize([256, 256]),
        # transform data into Tensor
        transforms.ToTensor(),
    ])

    transform_target = transforms.Compose([
        # resize image.
        transforms.Resize([256, 256]),
        # transform data into Tensor
        transforms.ToTensor()
    ])

    # Define dataset.
    train_set = dataset.jaffeDataset(train_set_target_path, train_set_input_path, lm_image_path, lm_csv_path, transform=transform)
    trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)


    # define model (generator and discriminator)
    G_l = generator.Pix2PixGenerator(label=torch.zeros((args.batch_size, 7))).to(device)
    optimizer_G_l = optim.Adam(G_l.parameters(), lr=args.lr, betas=args.betas)
    D_l = discriminator.Discriminator().to(device)
    optimizer_D_l = optim.Adam(D_l.parameters(), lr=args.lr, betas=args.betas)

    # define expression generator and discriminator.

    G_e = generator.Pix2PixGenerator(in_channels=6, label=torch.zeros((args.batch_size, 7))).to(device)
    optimizer_G_e = optim.Adam(G_e.parameters(), lr=args.lr, betas=args.betas)
    D_e = discriminator.Discriminator().to(device)
    optimizer_D_e = optim.Adam(D_e.parameters(), lr=args.lr, betas=args.betas)


    # resume training.
    # checkpoint_path = "./99_with_originalimg_id_5_facere_100_others_10_rm_normal_target_checkpoint.pth"
    # checkpoint = torch.load(checkpoint_path)
    # G.load_state_dict(checkpoint['G_state_dict'])
    # optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    # D.load_state_dict(checkpoint['D_state_dict'])
    # optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    # checkpoint_sche_path = "./99_with_originalimg_id_5_facere_100_others_10_rm_normal_target_sche_checkpoint.pth"
    # sche_checkpoint = torch.load(checkpoint_sche_path)

    # # Learning rate decay.
    scheduler_G_l = StepLR(optimizer_G_l, step_size=10, gamma=args.gamma)
    scheduler_D_l = StepLR(optimizer_D_l, step_size=10, gamma=args.gamma)
    scheduler_G_e = StepLR(optimizer_G_e, step_size=10, gamma=args.gamma)
    scheduler_D_e = StepLR(optimizer_D_e, step_size=10, gamma=args.gamma)
    # scheduler_G.load_state_dict(sche_checkpoint['G_scheduler_dict'])
    # scheduler_D.load_state_dict(sche_checkpoint['D_scheduler_dict'])
    for epoch in range(args.epochs + 1):
        train(args, G_l, D_l, G_e, D_l, device, trainloader, optimizer_G_l, optimizer_D_l, optimizer_G_e, optimizer_D_e, epoch)
        scheduler_G_l.step()
        scheduler_D_l.step()
        scheduler_G_e.step()
        scheduler_D_e.step()

        if epoch % 100 == 0:
            checkpoint_scheduler_path = './{}_{}_sche_checkpoint.pth'.format(epoch, training_description)
            torch.save({
                'G_l_scheduler_dict': scheduler_G_l.state_dict(),
                'D_l_scheduler_dict': optimizer_D_l.state_dict(),
                'G_e_scheduler_dict': scheduler_G_e.state_dict(),
                'D_e_scheduler_dict': optimizer_D_e.state_dict(),
            }, checkpoint_scheduler_path)

    end = time.time()
    print("Total training time is: " + str(end - start))
    # sys.stdout.close()


# Training
main()

