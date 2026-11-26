from PIL import Image
import os
import torch.utils.data as data
import torch
import numpy as np
import json
import cv2
import torchvision.transforms as transforms
import random
import datetime
import torch.nn.functional as F


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # numpy img: [L, C, H, W]
            img = torch.from_numpy(pic).permute(2, 3, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(
                pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255) if self.div else img.float()
        return img


class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        mode = img_group[0].mode            # 1116 PIL.mode 判断图象格式
        if mode == '1':
            img_group = [img.convert('L') for img in img_group]
            mode = 'L'
        if mode == 'L':
            return np.stack([np.expand_dims(x, 2) for x in img_group], axis=2)
            # 1116 掩膜到这里 list PIL [0,255] [w,h] -> list numpy uint8 [0,255] [h,w,c=1] -> numpy uint8 [0,255] [h,w,t,c=1]
        elif mode == 'RGB':
            if self.roll:
                return np.stack([np.array(x)[:, :, ::-1] for x in img_group],
                                axis=2)
            else:
                return np.stack(img_group, axis=2)      # 1116 帧到这里 numpy uint8 [0,255] [h,w,t,c]
        else:
            raise NotImplementedError(f"Image mode {mode}")


class RepeatDataset:
    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        return self.times * self._ori_len


def data_augmentation(cloud, mask, gt, flip_h=True, mir=True, rot=True):
    if flip_h and random.random() < 0.5:
        cloud = [i.transpose(Image.Transpose.FLIP_TOP_BOTTOM) for i in cloud]
        mask = [i.transpose(Image.Transpose.FLIP_TOP_BOTTOM) for i in mask]
        gt = [i.transpose(Image.Transpose.FLIP_TOP_BOTTOM) for i in gt]
    if mir and random.random() < 0.5:
        cloud = [i.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for i in cloud]
        mask = [i.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for i in mask]
        gt = [i.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for i in gt]
    if rot and random.random() < 0.5:
        cloud = [i.transpose(Image.Transpose.ROTATE_90) for i in cloud]
        mask = [i.transpose(Image.Transpose.ROTATE_90) for i in mask]
        gt = [i.transpose(Image.Transpose.ROTATE_90) for i in gt]
    return cloud, mask, gt


class TrainDataset(data.Dataset):
    """ random select ref frames and resize frames """
    def __init__(self, args: dict, debug=False):
        self.args = args
        self.num_local_frames = args['num_local']
        self.num_ref_frames = args['num_ref']
        self.size = self.w, self.h = (args['w'], args['h'])
        self.img_size = 640

        json_path = args['filepath']
        with open(json_path, 'r') as f:
            self.video_dict = json.load(f)
        self.video_names = list(self.video_dict.keys())
        if debug:
            self.video_names = self.video_names[:100]

        self.cloud_path = os.path.join(args['path'], 'cloud')
        self.mask_path = os.path.join(args['path'], 'mask')
        self.gt_path = os.path.join(args['path'], 'GT')
        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
        ])
        self.sup_type = args['sup_type']

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        item = self.load_item(index)
        return item

    def _sample_index(self, length, sample_length, num_ref_frame):
        complete_idx_set = list(range(length))
        pivot = random.randint(0, length - sample_length)
        local_idx = complete_idx_set[pivot:pivot + sample_length]
        remain_idx = list(set(complete_idx_set) - set(local_idx))
        ref_index = sorted(random.sample(remain_idx, num_ref_frame))
        return local_idx + ref_index

    def load_item(self, index):
        video_name = self.video_names[index]
        selected_index = self._sample_index(self.video_dict[video_name], self.num_local_frames, self.num_ref_frames)

        clouds, masks, gts, sup_mask = [], [], [], []
        cloud_path = os.path.join(self.cloud_path, video_name)
        mask_path = os.path.join(self.mask_path, video_name)
        gt_path = os.path.join(self.gt_path, video_name.split('_')[0])
        img_list = sorted(os.listdir(cloud_path))

        ch = random.randint(0, self.img_size - self.h)
        cw = random.randint(0, self.img_size - self.w)
        for idx in selected_index:
            cloud_img_path = os.path.join(cloud_path, img_list[idx])
            cloud = Image.open(cloud_img_path).convert('RGB')
            cloud = cloud.crop((cw, ch, cw + self.w, ch + self.h))
            clouds.append(cloud)

            mask_img_path = os.path.join(mask_path, img_list[idx])
            mask = Image.open(mask_img_path).convert('L')
            mask = mask.crop((cw, ch, cw + self.w, ch + self.h))
            masks.append(mask)

            gt_img_path = os.path.join(gt_path, img_list[idx])
            gt = Image.open(gt_img_path).convert('RGB')
            gt = gt.crop((cw, ch, cw + self.w, ch + self.h))
            gts.append(gt)

        clouds, masks, gts = data_augmentation(clouds, masks, gts)
        clouds_tensors = self._to_tensors(clouds) * 2.0 - 1.0
        masks_tensors = self._to_tensors(masks)
        gts_tensors = self._to_tensors(gts) * 2.0 - 1.0
        sup_tensors = F.max_pool2d(masks_tensors.float(), kernel_size=4, stride=4)

        return clouds_tensors, masks_tensors, gts_tensors, sup_tensors, video_name


class EvalDataset(data.Dataset):
    def __init__(self, args):
        self.args = args
        with open(args['filepath'], 'r') as f:
            self.video_dict = json.load(f)
        self.video_names = list(self.video_dict.keys())

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
        ])
        self.mode = args['mode']
        self.cloud_path = os.path.join(args['path'], 'cloud')
        self.mask_type = args['mask_type']
        self.gt_path = os.path.join(args['path'], 'GT')
        if self.mask_type == 'coarse':
            self.mask_path = os.path.join(args['path'], 'coarse_mask')
        elif self.mask_type == 'dilate' or 'refine':
            self.mask_path = os.path.join(args['path'], 'mask')

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        item = self.load_item(index)
        return item

    def load_item(self, index):
        video_name = self.video_names[index]
        ref_index = list(range(self.video_dict[video_name]))
        clouds, masks, gts = [], [], []
        cloud_path = os.path.join(self.cloud_path, video_name)
        mask_path = os.path.join(self.mask_path, video_name)
        gt_path = os.path.join(self.gt_path, video_name.split('_')[0])
        img_list = sorted(os.listdir(cloud_path))

        for idx in ref_index:
            cloud_img_path = os.path.join(cloud_path, img_list[idx])
            clouds.append(Image.open(cloud_img_path).convert('RGB'))

            gt_img_path = os.path.join(gt_path, img_list[idx])
            gts.append(Image.open(gt_img_path).convert('RGB'))

            mask_img_path = os.path.join(mask_path, img_list[idx])
            mask = Image.open(mask_img_path).convert('L')

            if self.mask_type == 'dilate':
                mask = np.asarray(mask)
                m = np.array(mask > 0).astype(np.uint8)
                m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=4)
                mask = Image.fromarray(m * 255)
            masks.append(mask)

        frame_PIL = [np.array(f).astype(np.uint8) for f in gts]
        clouds_tensors = self._to_tensors(clouds) * 2.0 - 1.0
        masks_tensors = self._to_tensors(masks)

        return clouds_tensors, masks_tensors, video_name, frame_PIL


class TestDataset(data.Dataset):
    def __init__(self, args):
        self.args = args
        with open(args['filepath'], 'r') as f:
            self.video_dict = json.load(f)
        self.video_names = list(self.video_dict.keys())

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
        ])
        self.mode = args['mode']
        self.cloud_path = os.path.join(args['path'], 'cloud')

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        item = self.load_item(index)
        return item

    def load_item(self, index):
        video_name = self.video_names[index]
        ref_index = list(range(self.video_dict[video_name]))
        clouds, masks, gts = [], [], []
        cloud_path = os.path.join(self.cloud_path, video_name)
        cloud_list = sorted(os.listdir(cloud_path))

        for idx in ref_index:
            cloud_img_path = os.path.join(cloud_path, cloud_list[idx])
            clouds.append(Image.open(cloud_img_path).convert('RGB'))

        clouds_tensors = self._to_tensors(clouds) * 2.0 - 1.0
        return clouds_tensors, [], video_name, []
