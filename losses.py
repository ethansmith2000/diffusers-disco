

import torch

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import math
import torchvision.transforms.functional as TF


is_colab=False
padargs = {}

ceiling = 1.0


def resize(image, size, mode):
    orig_dtype = image.dtype
    image=image.float()
    image = torch.nn.functional.interpolate(image, size=size, mode=mode, align_corners=False)
    image=image.to(orig_dtype)
    return image


class MakeCutouts(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pixel_values, num_cutouts, cut_size, cut_power=1.0):
        sideY, sideX = pixel_values.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, cut_size)
        cutouts = []
        for _ in range(num_cutouts):
            size = int(torch.rand([]) ** cut_power * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = pixel_values[:, :, offsety : offsety + size, offsetx : offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, cut_size))
        return torch.cat(cutouts)


class MakeCutoutsES(nn.Module):
    def __init__(self, cut_size,
                 randoffsetx,
                 randoffsety,
                 randsize,
                 randsize2,
                 timestep,
                 Overview=4,
                 InnerCrop = 0,
                 IC_Size_Pow=0.5,
                 IC_Grey = 0.2,
                 overview_gray = 0.1,
                 ov_proportion = 0.5,
                 floor = 0.1,
                 all_models_same_cuts = 0,

                 col_jitter = 0.1,
                    pixel_jitter = 0.1,
                 percent_stop_affine = 0.5,
                    degree_tilt = 10,
                    affine = True,
                    hflip = True,
                    perspective = True,

                adaptive_weight = False,
                 overview_type = "jaxy_overviews",
                 skip_augs=False,
                 resize_type = "bilinear",

                debug=False
                 ):
        super().__init__()
        self.cut_size = cut_size
        self.Overview = Overview
        self.InnerCrop = InnerCrop
        self.IC_Size_Pow = IC_Size_Pow
        self.IC_Grey = IC_Grey
        self.overview_gray = overview_gray
        self.ov_proportion = ov_proportion
        self.floor = floor
        self.use_floor = floor > 0

        self.debug = debug

        self.overview_type = overview_type
        self.skip_augs = skip_augs
        self.adaptive_weight = adaptive_weight
        self.resize_type = resize_type

        self.randoffsetx = randoffsetx
        self.randoffsety = randoffsety
        self.randsize = randsize
        self.randsize2 = randsize2
        self.sizes = []

        self.total_num_cuts = self.Overview + self.InnerCrop

        # augmentations for inner cuts
        self.ic_augs = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Lambda(lambda x: x + torch.randn_like(x) * pixel_jitter),
            T.ColorJitter(brightness=col_jitter, contrast=col_jitter, saturation=col_jitter, hue=col_jitter),
        ])

        # augmentations for overview cuts
        tmpaugs = [
            T.Lambda(lambda x: x + torch.randn_like(x) * pixel_jitter),
            T.ColorJitter(brightness=col_jitter, contrast=col_jitter, saturation=col_jitter, hue=col_jitter),
        ]
        if (hflip ==True):
            tmpaugs.append(T.RandomHorizontalFlip(p=0.5))
        if (perspective == True):
            tmpaugs.append(T.RandomPerspective(distortion_scale=0.18, p=0.35))
        if affine == True and timestep > percent_stop_affine:
            tmpaugs.append(T.RandomAffine(degrees=degree_tilt, translate=(0.05, 0.05),  interpolation = T.InterpolationMode.BILINEAR))

        self.gray_transform = T.Grayscale(3)

        self.ov_augs = T.Compose(tmpaugs)
        self.all_models_same_cuts = all_models_same_cuts


    def do_augs(self, cutouts, cutout, size, type="full"):

        gray_cond = self.overview_gray if (type == "overview" or type == "full") else self.IC_Grey
        augs = self.ov_augs if (type == "overview" or type == "full") else self.ic_augs

        # randomly gray
        if (torch.rand([]) < gray_cond):
            cutout = self.gray_transform(cutout)

        # apply augmentations
        orig_dtype = cutout.dtype
        cutout = cutout.float()
        if not self.skip_augs:
            cutout = augs(cutout).to(orig_dtype)
            cutouts.append(cutout)
        else:
            cutouts.append(cutout.to(orig_dtype))


        if self.adaptive_weight:
            self.sizes.append(size)

        return cutouts


    def cutout_debug(self, input):
        if is_colab:
            TF.to_pil_image(input.clamp(0, 1).squeeze(0)).save("/content/cutout_debug.jpg", quality=99)
        else:
            TF.to_pil_image(input.clamp(0, 1).squeeze(0)).save("cutout_debug.jpg", quality=99)

    def forward(self, input):
        cutouts = []
        counter = 0
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        floor_size = min_size + (max_size - min_size) * self.floor
        # output_shape = [1, 3, self.cut_size, self.cut_size]
        output_shape = (self.cut_size, self.cut_size)
        pad_input = F.pad(input, (
        (sideY - max_size) // 2, (sideY - max_size) // 2, (sideX - max_size) // 2, (sideX - max_size) // 2), **padargs)
        bordery = ((sideY - max_size) // 2) * 2
        borderx = ((sideX - max_size) // 2) * 2
        border = int(((sideX * sideY) ** 0.5) * 0.1)

        # set random values
        if self.all_models_same_cuts:
            randsize = self.randsize
            randsize2 = self.randsize2
            randoffsetx = self.randoffsetx
            randoffsety = self.randoffsety
            randborderx = [int(self.randoffsetx[i] * border) for i in range(self.total_num_cuts)]
            randbordery = [int(self.randoffsety[i] * border) for i in range(self.total_num_cuts)]
        else:
            randsize = torch.rand(self.total_num_cuts).tolist()
            randsize2 = torch.rand(self.total_num_cuts).tolist()
            randoffsetx = torch.rand(self.total_num_cuts).tolist()
            randoffsety = torch.rand(self.total_num_cuts).tolist()
            randborderx = [int(randoffsetx[i] * border) for i in range(self.total_num_cuts)]
            randbordery = [int(randoffsety[i] * border) for i in range(self.total_num_cuts)]

        if self.Overview > 0:

            # not full screen overview cuts
            for j in range(int(math.ceil(self.Overview * self.ov_proportion))):

                if self.overview_type == "jaxy_overviews":
                    # import pdb
                    # pdb.set_trace()
                    my_pad = F.pad(input, (
                    math.ceil(bordery * randsize2[counter]) + 20 + math.ceil(randbordery[counter] * randsize[counter]),
                    math.floor(bordery * (1 - randsize2[counter])) + 20 + math.floor(
                        (border - randbordery[counter]) * randsize[counter]),
                    math.ceil(borderx * randsize2[counter]) + 20 + math.ceil(randborderx[counter] * randsize[counter]),
                    math.floor(borderx * (1 - randsize2[counter])) + 20 + math.floor(
                        (border - randborderx[counter]) * randsize[counter])), **padargs)
                    cutout = resize(my_pad, size=output_shape, mode=self.resize_type)
                elif self.overview_type == "jaxy_overviews2":
                    my_pad = F.pad(input, (
                    (sideY - max_size) // 2 + 20 + math.ceil(randbordery[counter] * randsize[counter]),
                    (sideY - max_size) // 2 + 20 + math.floor((80 - randbordery[counter]) * randsize[counter]),
                    (sideX - max_size) // 2 + 20 + math.ceil(randborderx[counter] * randsize[counter]),
                    (sideX - max_size) // 2 + 20 + math.floor((80 - randborderx[counter]) * randsize[counter])), **padargs)
                    cutout = resize(my_pad, size=output_shape, mode=self.resize_type)
                elif self.overview_type == "zoom_out_overviews":
                    my_pad = F.pad(input, ((sideY - max_size) // 2 + 10 + math.ceil(border * randsize[counter]),
                                           (sideY - max_size) // 2 + 10 + math.floor(border *randsize[counter]),
                                           (sideX - max_size) // 2 + 10 + math.ceil(border * randsize[counter]),
                                           (sideX - max_size) // 2 + 10 + math.floor(border * randsize[counter])),
                                   **padargs)
                    cutout =resize(my_pad, size=output_shape, mode=self.resize_type)
                else:
                    cutout = resize(pad_input, size=output_shape, mode=self.resize_type)

                #TODO maybe augs should be done before resize?
                cutouts = self.do_augs(cutouts, cutout, max(sideX, sideY) ** 2, type="full")

                counter += 1

            for k in range(int(math.floor(self.Overview * (1 - self.ov_proportion)))):
                if (randsize[counter] < 0.8):
                    size = max_size
                else:
                    size = int((max_size - min_size) * 0.7 + min_size)

                offsetx = int((randoffsetx[counter]) * (sideX - size + 1))
                offsety = int((randoffsety[counter]) * (sideY - size + 1))
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]

                cutout = resize(cutout, size=output_shape, mode=self.resize_type)

                cutouts = self.do_augs(cutouts, cutout, size ** 2, type="overview")

                counter += 1

            if self.debug:
                self.cutout_debug(cutouts[0])

        if self.InnerCrop > 0:
            for i in range(self.InnerCrop):
                if self.use_floor:
                    size = int(randsize[counter] ** self.IC_Size_Pow * ceiling * (
                                max_size - floor_size) + floor_size)
                else:
                    size = int(
                        randsize[counter] ** self.IC_Size_Pow * ceiling * (max_size - min_size) + min_size)
                offsetx = int((randoffsetx[counter]) * (sideX - size + 1))
                offsety = int((randoffsety[counter]) * (sideY - size + 1))
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]

                cutout = resize(cutout, size=output_shape, mode=self.resize_type)

                cutouts = self.do_augs(cutouts, cutout, size ** 2, type="innercut")

                counter += 1

            if self.debug:
                self.cutout_debug(cutouts[-1])

        cutouts = torch.cat(cutouts)
        return cutouts


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def symm_loss(im, lpm):
    h = int(im.shape[3] / 2)
    h1, h2 = im[:, :, :, :h], im[:, :, :, h:]
    h2 = TF.hflip(h2)
    return lpm(h1, h2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff ** 2 + y_diff ** 2).mean([1, 2, 3])


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])