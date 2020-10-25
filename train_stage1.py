from discriminator import Discriminator
from warp_model import WarpModel
from warp_dataset import WarpDataset
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
from PIL import Image
from tqdm import tqdm
def denorm(tensor, mean=[0.06484050184440379, 0.06718090599394404, 0.07127327572275131],
           std=[0.2088075459038679, 0.20012519201951368, 0.23498672043315685], clamp=True, inplace=False):
    if not inplace:
        tensor = tensor.clone()

    def unnormalize_1(ten, men, st):
        for t, m, s in zip(ten, men, st):
            t.mul_(s).add_(m)
            if clamp:
                t.clamp_(0, 1)

    if tensor.shape == 4:
        # then we have batch size in front or something
        for t in tensor:
            unnormalize_1(t, mean, std)
    else:
        unnormalize_1(tensor, mean, std)

    return tensor
label_colours = [(0,0,0)
                 # 0=Background
    ,(128,0,0),(255,0,0),(0,85,0),(170,0,51),(255,85,0)
                 # 1=Hat,  2=Hair,    3=Glove, 4=Sunglasses, 5=UpperClothes
    ,(0,0,85),(0,119,221),(85,85,0),(0,85,85),(85,51,0)
                 # 6=Dress, 7=Coat, 8=Socks, 9=Pants, 10=Jumpsuits
    ,(52,86,128),(0,128,0),(0,0,255),(51,170,221),(0,255,255)
                 # 11=Scarf, 12=Skirt, 13=Face, 14=LeftArm, 15=RightArm
    ,(85,255,170),(170,255,85),(255,255,0),(255,170,0)]
# 16=LeftLeg, 17=RightLeg, 18=LeftShoe, 19=RightShoe


# take out sunglasses
label_colours = label_colours[:4] + label_colours[5:]
n_classes = 19
def decode_cloth_labels(pt_tensor, num_images=-1, num_classes=n_classes):
    """Decode batch of segmentation masks.
    AJ comment: Converts the tensor into a RGB image.
    Args:
      as_tf_order: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    # change to H x W x C order
    tf_order = pt_tensor.permute(0, 2, 3, 1)
    argmax = tf_order.argmax(dim=-1, keepdim=True)
    mask = argmax.cpu().numpy()

    n, h, w, c = mask.shape
    if num_images < 0:
        num_images = n
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        # AJ: this enumerates the "rows" of the image (I think)
        for j_, j in enumerate(mask[i, :, :, 0]):
            for k_, k in enumerate(j):
                if k < n_classes:
                    pixels[k_,j_] = label_colours[k]
        outputs[i] = np.array(img)

    # convert back to tensor. effectively puts back into range [0,1]
    back_to_pt = torch.from_numpy(outputs).permute(0, 3, 1, 2)
    return back_to_pt
def train():
    batch_size = 16
    warp_model = WarpModel().cuda()
    discriminator = Discriminator(22).cuda()
    optimizer_W = optim.Adam(warp_model.parameters(), 0.0001, (0.5, 0.999))
    optimizer_D = optim.SGD(discriminator.parameters(), lr=0.0001, momentum=0.9, nesterov=True)
    warp_dataset = WarpDataset()
    warp_loader = DataLoader(warp_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    real_labels = torch.ones(size=(batch_size, 1, 14, 14), requires_grad=False).cuda()
    fake_labels = torch.zeros(size=(batch_size, 1, 14, 14), requires_grad=False).cuda()
    l2_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    num_epoch = 10
    for epoch in range(num_epoch):
        tqdm_loader = tqdm(iter(warp_loader), leave=True, total=len(warp_loader))
        for i, data in enumerate(tqdm_loader):
            bodys = data["bodys"].cuda()
            print(bodys.size())
            #save_image(denorm(bodys.cpu())[0], "./Result/{}_Body.png".format(epoch))
            inputs = data["input_cloths"].cuda()
            print(inputs.size())
            targets = data["target_cloths"].cuda()
            print(targets.size())
            exit()
            warp_result = warp_model(bodys, inputs)
            # Training Discriminator
            optimizer_D.zero_grad()
            conditioned_fake = torch.cat((bodys, warp_result), dim=1)
            pred_fake = discriminator(conditioned_fake.detach())
            loss_d_fake = l2_loss(pred_fake, fake_labels)

            conditioned_real = torch.cat((bodys, targets), dim=1)
            pred_real = discriminator(conditioned_real)
            loss_d_real = l2_loss(pred_real, real_labels)
            loss_D = 0.5 * (loss_d_fake + loss_d_real)
            loss_D.backward()
            optimizer_D.step()

            # Training WarpModel
            optimizer_W.zero_grad()
            fool_d = discriminator(conditioned_fake.detach())
            loss_fool = l2_loss(fool_d, real_labels)
            loss_ce = ce_loss(warp_result, torch.argmax(targets, dim=1))
            loss_W = loss_fool + loss_ce
            loss_W.backward()
            optimizer_W.step()
            # save_image(decode_cloth_labels(inputs, num_images=1)[0], "./Result/{}_Input.png".format(epoch))
            # save_image(decode_cloth_labels(bodys, num_images=1)[0], "./Result/{}_Body.png".format(epoch))
            # save_image(decode_cloth_labels(targets, num_images=1)[0], "./Result/{}_target.png".format(epoch))
            # save_image(decode_cloth_labels(warp_result, num_images=1)[0], "./Result/{}_result.png".format(epoch))
            # exit()

        print("Epoch {0}\tD Loss:{1}\tWarp Loss:{2}".format(epoch, loss_D.item(), loss_W.item()))
        save_image(decode_cloth_labels(inputs, num_images=1)[0], "./Result/{}_Input.png".format(epoch))
        save_image(decode_cloth_labels(bodys, num_images=1)[0], "./Result/{}_Body.png".format(epoch))
        save_image(decode_cloth_labels(targets, num_images=1)[0], "./Result/{}_target.png".format(epoch))
        save_image(decode_cloth_labels(warp_result, num_images=1)[0], "./Result/{}_result.png".format(epoch))



if __name__ == "__main__":
    train()