import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import sys
import datetime
from loggers import Logger
import numpy as np
import random
from dataset import EvalDataset, TestDataset
import json
import cv2
import importlib
from metrics import f_boundary, jaccard
from metrics.psnr import calc_psnr_and_ssim
from frame_select import get_ref_index_uniform

parser = argparse.ArgumentParser(description='PyTorch CloudHandler Example')
parser.add_argument('--config', type=str, default='config/eval_model.json')
# parser.add_argument('--config', type=str, default='config/test_model.json')
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
opt = parser.parse_args()


def set_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise Exception('No Gpu found, please run with gpu')
    else:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = True

def save_results(save_frame_path, comp_frames, pred_masks):
    os.makedirs(save_frame_path, exist_ok=False)
    cr_path = os.path.join(save_frame_path, 'removal')
    os.makedirs(cr_path, exist_ok=False)
    cd_path = os.path.join(save_frame_path, 'detection')
    os.makedirs(cd_path, exist_ok=False)

    for i, frame in enumerate(comp_frames):
        cv2.imwrite(os.path.join(cr_path, str(i).zfill(8) + '.png'),
                    cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))
    for i, pred in enumerate(pred_masks):
        cv2.imwrite(os.path.join(cd_path, str(i).zfill(8) + '.png'), (pred * 255.0).astype(np.uint8))

def test_model(netG, config, mode = 'offline', iter=None, writer=None):
    print('============ Testing Stage ============')
    seed = config.get('seed', random.randint(1, 10000))
    set_seed(seed)

    assert mode == 'offline' or 'online'
    netG.eval()

    test_dataset = EvalDataset(config['test_data_loader']) if config['test_data_loader']['mode'] == 'eval' \
        else (TestDataset(config['test_data_loader']))
    test_loader = DataLoader(dataset=test_dataset, num_workers=0, batch_size=1, shuffle=False)
    print('TestSet Length: ', len(test_dataset), 'TestLoader Length: ', len(test_loader))

    if config['test_data_loader']['mode'] == 'eval':
        total_frame_psnr, total_frame_ssim = [], []
        total_J, total_F, total_JFmean = [], [], []

    with torch.no_grad():
        for index, data in enumerate(test_loader):
            frames, masks, video_name, frames_PIL = data
            video_length = frames.size(1)

            if config['test_data_loader']['mode'] == 'eval':
                ori_frames = [
                    frames_PIL[i].squeeze().numpy() for i in range(video_length)
                ]
                gt_masks = [
                    masks[:, i, :, :, :].permute(0, 2, 3, 1).numpy().squeeze(0).astype(np.float32) for i in
                    range(video_length)
                ]

            frames = frames.cuda()
            comp_frames = [None] * video_length
            pred_masks = [None] * video_length

            neighbor_stride = config['test_data_loader']['neighbor_stride']
            for f in range(0, video_length, neighbor_stride):
                if config['test_data_loader']['overlap']:
                    neighbor_ids = [i for i in
                                    range(max(0, f - neighbor_stride), min(video_length, f + neighbor_stride + 1))]
                else:
                    neighbor_ids = [i for i in range(f, min(video_length, f + neighbor_stride))]

                if config['test_data_loader']['num_ref'] != 0:
                    ref_ids = get_ref_index_uniform(neighbor_ids, video_length, config)
                else:
                    ref_ids = []
                # print(neighbor_ids + ref_ids)
                selected_imgs = frames[:1, neighbor_ids + ref_ids, :, :, :]
                cd_out, cr_out, _ = netG(selected_imgs, num_ref=config['test_data_loader']['num_ref'])

                pred_img = (cr_out.cpu() + 1) / 2
                pred_img = pred_img.permute(0, 2, 3, 1).numpy() * 255
                cd_out = cd_out.data.cpu().permute(0, 2, 3, 1)

                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_img[i]).astype(np.uint8)

                    pred_mask = cd_out[i].sigmoid().numpy()
                    pred_mask = (pred_mask - pred_mask.min()) / (pred_mask.max() - pred_mask.min() + 1e-8)
                    pred_mask[np.where(pred_mask >= 0.5)] = 1
                    pred_mask[np.where(pred_mask < 0.5)] = 0

                    if pred_masks[idx] is None:
                        pred_masks[idx] = np.array(pred_mask).astype(np.float32)
                    else:
                        pred_masks[idx] = pred_masks[idx] * 0.5 + np.array(pred_mask).astype(np.float32) * 0.5

                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else:
                        comp_img = comp_frames[idx].astype(
                            np.float32) * 0.5 + img.astype(np.float32) * 0.5
                        comp_frames[idx] = comp_img.astype(np.uint8)

            if mode == 'offline' and config['test_data_loader']['mode'] != 'eval':
                print('===> Scene {} has been generated'.format(video_name))

            if config['test_data_loader']['mode'] == 'eval':
                cur_video_psnr, cur_video_ssim = [], []
                curv_J, curv_F, curv_JFmean = [], [], []

                for ori, comp in zip(ori_frames, comp_frames):
                    psnr, ssim = calc_psnr_and_ssim(ori, comp)
                    cur_video_psnr.append(psnr)
                    cur_video_ssim.append(ssim)
                    total_frame_psnr.append(psnr)
                    total_frame_ssim.append(ssim)
                    # print(psnr, ssim)

                for gt, pred in zip(gt_masks, pred_masks):
                    J_score = jaccard.db_eval_iou(annotation=gt[:, :, 0], segmentation=pred[:, :, 0])
                    F_score = f_boundary.db_eval_boundary(foreground_mask=gt[:, :, 0], gt_mask=pred[:, :, 0])
                    JFmean = (J_score + F_score) / 2
                    curv_J.append(J_score)
                    curv_F.append(F_score)
                    curv_JFmean.append(JFmean)
                    total_J.append(J_score)
                    total_F.append(F_score)
                    total_JFmean.append(JFmean)

                cur_psnr = sum(cur_video_psnr) / len(cur_video_psnr)
                cur_ssim = sum(cur_video_ssim) / len(cur_video_ssim)
                cur_J = sum(curv_J) / len(curv_J)
                cur_F = sum(curv_F) / len(curv_F)
                cur_JFmean = sum(curv_JFmean) / len(curv_JFmean)

                if mode == 'online' and writer is not None:
                    writer.add_scalar('scene_J/{}'.format(video_name), cur_J, iter)
                    writer.add_scalar('scene_F/{}'.format(video_name), cur_F, iter)
                    writer.add_scalar('scene_JFMean/{}'.format(video_name), cur_JFmean, iter)
                    writer.add_scalar('scene_PSNR/{}'.format(video_name), cur_psnr, iter)
                    writer.add_scalar('scene_SSIM/{}'.format(video_name), cur_ssim, iter)

                print(
                    f'[{index + 1}/{len(test_loader)}] Name: {str(video_name)} | PSNR/SSIM: {cur_psnr:.4f}/{cur_ssim:.4f} '
                    f' | J/F/JFMean: {cur_J:.4f}/{cur_F:.4f}/{cur_JFmean:.4f}')

            if mode == 'online' and config['test_data_loader']['save_im_results']:
                save_frame_path = os.path.join('im_results', config['name'], str(iter), video_name[0])
                save_results(save_frame_path, comp_frames, pred_masks)

            elif mode == 'offline' and config['test_data_loader']['save_results']:
                save_frame_path = os.path.join('results', config['name'], 'pic', video_name[0])
                save_results(save_frame_path, comp_frames, pred_masks)


        if config['test_data_loader']['mode'] == 'eval':
            avg_frame_psnr = sum(total_frame_psnr) / len(total_frame_psnr)
            avg_frame_ssim = sum(total_frame_ssim) / len(total_frame_ssim)
            avg_J = sum(total_J) / len(total_J)
            avg_F = sum(total_F) / len(total_F)
            avg_JFmean = sum(total_JFmean) / len(total_JFmean)

            if mode == 'online' and writer is not None and iter is not None:
                writer.add_scalar('iter_PSNR', avg_frame_psnr, iter)
                writer.add_scalar('iter_SSIM', avg_frame_ssim, iter)
                writer.add_scalar('iter_J', avg_J, iter)
                writer.add_scalar('iter_F', avg_F, iter)
                writer.add_scalar('iter_JFmean', avg_JFmean, iter)
            print(
                'Finish evaluation... Average Frame PSNR SSIM J F JFMean: 'f'{avg_frame_psnr:.4f} {avg_frame_ssim:.4f} '
                f'{avg_J:.4f} {avg_F:.4f} {avg_JFmean:.4f}')
    print('============  Finish evaluation  ============')
    if config['test_data_loader']['mode'] == 'eval':
        return avg_frame_psnr, avg_frame_ssim


def main():
    config = json.load(open(opt.config))
    save_test_log = 'results/' + config['name'] + '/log'
    sys.stdout = Logger(os.path.join(save_test_log, systime + '_test_' + config['name'] + '.txt'))
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_devices']
    print(config)

    net = importlib.import_module('model.' + config['model']['net'])
    netG = net.InpaintGenerator()
    print('===> {} model has been initialized'.format(config['model']['net']))
    print("Model size: {:.5f}M".format(sum(p.numel() for p in netG.parameters()) / 1e6))
    netG = netG.cuda()

    print('===> load pretrained model')
    if config['modelpath'] and os.path.exists(config['modelpath']):
        print('===> pretrained model {} is load'.format(config['modelpath']))
        data = torch.load(config['modelpath'])
        netG.load_state_dict(data)
    else:
        print('pretrain model is not exists. train with initial')

    test_model(netG, config, mode='offline', iter=None, writer=None)


if __name__ == '__main__':
    main()
