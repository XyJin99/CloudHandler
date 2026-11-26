import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
import datetime
import sys
from loggers import Logger
from torch.utils.data import DataLoader
from dataset import RepeatDataset, TrainDataset
from torch.optim.lr_scheduler import MultiStepLR
from tensorboardX import SummaryWriter
import numpy as np
import random
import json
import importlib
from loss import AdversarialLoss
from test import test_model

parser = argparse.ArgumentParser(description='PyTorch CloudHandler Example')
parser.add_argument('--config', type=str, default='config/train_model.json')
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

def compute_generator_loss(cd_out, cr_out, im_masks, frames, masks, gts, sup_masks, config, netD=None):
    b, t, c, h, w = frames.size()

    bce_loss = nn.BCEWithLogitsLoss().cuda()
    sup_loss = bce_loss(im_masks, sup_masks.view(b * t, -1, h // 4, w // 4))
    bce_out = bce_loss(cd_out, masks.view(b * t, -1, h, w))
    loss = sup_loss + bce_out

    pred_imgs = cr_out.view(b, -1, c, h, w)
    l1_loss = nn.L1Loss().cuda()

    sep_mask = False
    if torch.mean(masks) == 0 or torch.mean(1 - masks) == 0:
        hole_loss = l1_loss(pred_imgs, gts)
        loss += hole_loss
        sep_mask = True
    else:
        hole_loss = l1_loss(pred_imgs * masks, gts * masks)
        hole_loss = hole_loss / torch.mean(masks) * config['losses']['hole_weight']

        valid_loss = l1_loss(pred_imgs * (1 - masks), gts * (1 - masks))
        valid_loss = valid_loss / torch.mean(1 - masks) * config['losses']['valid_weight']
        loss += hole_loss + valid_loss

    log_dict = {
        'sup_loss': sup_loss.item(),
        'bce_loss': bce_out.item(),
        'hole_loss': hole_loss.item(),
        'valid_loss': valid_loss.item() if not sep_mask else 0
    }

    if netD is not None:
        adversarial_loss = AdversarialLoss('hinge').cuda()
        gen_vid_feat = netD(pred_imgs)
        gan_loss = adversarial_loss(gen_vid_feat, True, False)
        gan_loss = gan_loss * config['losses']['adversarial_weight']
        loss += gan_loss
        log_dict['gan_loss'] = gan_loss.item()

    return loss, log_dict

def compute_discriminator_loss(netD, real_imgs, fake_imgs, config):
    adversarial_loss = AdversarialLoss(type=config['losses']['GAN_LOSS']).cuda()
    real_feat = netD(real_imgs)
    fake_feat = netD(fake_imgs.detach())
    dis_real_loss = adversarial_loss(real_feat, True, True)
    dis_fake_loss = adversarial_loss(fake_feat, False, True)
    dis_loss = (dis_real_loss + dis_fake_loss) / 2
    return dis_loss, dis_real_loss.item(), dis_fake_loss.item()

def main():
    config = json.load(open(opt.config))

    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_devices']
    seed = config.get('seed', random.randint(1, 10000))
    set_seed(seed)

    writer_path = 'experiments/' + config['name'] + '/tb_logger'
    writer = SummaryWriter(writer_path)

    save_train_log = 'experiments/' + config['name'] + '/log'
    sys.stdout = Logger(os.path.join(save_train_log, systime + 'train_' + config['name'] + '.txt'))
    print('Set random seed to ', seed)

    print(config)
    print('===> Loading Datasets')
    assert config['train_data_loader']['ref_type'] == 'uniform' or 'select'

    train_dataset = TrainDataset(config['train_data_loader'])
    print('uniform training using fixed masks')

    iterations = config['trainer'].get('iterations', False)
    train_dataset = RepeatDataset(train_dataset, config['trainer'].get('enlarge', 1000))
    train_loader = DataLoader(dataset=train_dataset, num_workers=config['trainer']['num_workers'],
                              batch_size=config['trainer']['batch_size'],
                              shuffle=True)
    if len(train_loader) < config['trainer']['iterations']:
        raise Exception(
            'data enlarge is not enough. Get train_loader length={},nIter={}'.format(len(train_loader), iterations))

    print('TrainLoader Length: ', len(train_loader))
    net = importlib.import_module('model.' + config['model']['net'])
    netG = net.InpaintGenerator()
    print('===> {} model has been initialized'.format(config['model']['net']))
    print(netG)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in netG.parameters()) / 1e6))
    netG = netG.cuda()

    if not config['model']['no_dis']:
        netD = net.Discriminator(in_channels=3, use_sigmoid=config['losses']['GAN_LOSS'] != 'hinge')
        print('===> Discriminator has been initialized')
        print(netD)
        print("Model size: {:.5f}M".format(sum(p.numel() for p in netD.parameters()) / 1e6))
        netD = netD.cuda()

    print('start training...')
    Startiter, BestPSNR, BestIter = 0, 0, 0
    optimizer_G = torch.optim.Adam(netG.parameters(), lr=config['trainer']['lr'],
                                    betas=(config['trainer']['beta1'], config['trainer']['beta2']))
    scheduler_G = MultiStepLR(optimizer_G, milestones=config['trainer']['scheduler']['milestones'],
                               gamma=config['trainer']['scheduler']['gamma'])

    if not config['model']['no_dis']:
        optimizer_D = torch.optim.Adam(netD.parameters(), lr=config['trainer']['lr'],
                                       betas=(config['trainer']['beta1'], config['trainer']['beta2']))
        scheduler_D = MultiStepLR(optimizer_D, milestones=config['trainer']['scheduler']['milestones'],
                               gamma=config['trainer']['scheduler']['gamma'])

    if config['trainer']['resume']:
        print('resuming...')
        data = torch.load(config['trainer']['gen_path'], map_location='cuda:0')
        netG.load_state_dict(data)
        if not config['model']['no_dis']:
            data = torch.load(config['trainer']['dis_path'], map_location='cuda:0')
            netD.load_state_dict(data)

        data_opt = torch.load(config['trainer']['opt_path'], map_location='cuda:0')
        optimizer_G.load_state_dict(data_opt['optimG'])
        scheduler_G.load_state_dict(data_opt['scheG'])
        if not config['model']['no_dis']:
            optimizer_D.load_state_dict(data_opt['optimD'])
            scheduler_D.load_state_dict(data_opt['scheD'])
        Startiter = data_opt['iteration']

    Loss_summary = {'dis_loss': 0, 'gan_loss': 0, 'hole_loss': 0, 'valid_loss': 0, 'gen_loss': 0,
                    'bce_loss': 0, 'sup_loss': 0}

    for iter, data in enumerate(train_loader):
        iter = iter + Startiter
        if iter + 1 > config['trainer']['iterations']:
            break

        t2 = time.time()
        frames, masks, gts, sup_masks = data[0], data[1], data[2], data[3]
        b, t, c, h, w = frames.size()
        frames, masks, gts, sup_masks = frames.cuda(), masks.cuda(), gts.cuda(), sup_masks.cuda()

        netG.train()
        netD.train()

        t0 = time.time()
        cd_out, cr_out, im_masks = netG(frames, num_ref=config['train_data_loader']['num_ref'])
        t1 = time.time()

        if not config['model']['no_dis']:
            optimizer_D.zero_grad()
            dis_loss, dis_real, dis_fake = compute_discriminator_loss(netD, gts, cr_out.view(b, -1, c, h, w), config)
            dis_loss.backward()
            optimizer_D.step()
            scheduler_D.step()

            writer.add_scalar('loss/dis/dis_vid_fake', dis_fake, iter)
            writer.add_scalar('loss/dis/dis_vid_real', dis_real, iter)
            writer.add_scalar('loss/dis/dis_loss', dis_loss.item(), iter)
            Loss_summary['dis_loss'] += dis_loss.item()

        optimizer_G.zero_grad()
        gen_loss, log_dict = compute_generator_loss(
            cd_out=cd_out,
            cr_out=cr_out,
            im_masks=im_masks,
            frames=frames,
            masks=masks,
            gts=gts,
            sup_masks=sup_masks,
            config=config,
            netD=netD if not config['model']['no_dis'] else None)

        if torch.isfinite(gen_loss):
            gen_loss.backward()
            optimizer_G.step()
            scheduler_G.step()
        else:
            print(f"[Warning] NaN or Inf in generator loss at iter {iter}")

        writer.add_scalar('loss/gen/gen_loss', gen_loss.item(), iter)
        Loss_summary['gen_loss'] += gen_loss.item()
        for k, v in log_dict.items():
            writer.add_scalar(f'loss/gen/{k}', v, iter)
            Loss_summary[k] += v

        t3 = time.time()
        if (iter+1) % (config['trainer']['log_freq']) == 0:
            for key, value in Loss_summary.items():
                Loss_summary[key] = value / config['trainer']['log_freq']
            print("===> iter[{}]/{}): Hole_Loss: {:.4f}  Valid_Loss: {:.4f}  bce_Loss: {:.4f}  sup_Loss: {:.4f}  "
                  "Gan_Loss: {:.4f}  Gen_Loss: {:.4f}  Dis_Loss: {:.4f}  lr: {:.6E}  Timer: {:.4f}/{:.4f} sec."
                  .format(iter+1, config['trainer']['iterations'], Loss_summary['hole_loss'],
                          Loss_summary['valid_loss'], Loss_summary['bce_loss'], Loss_summary['sup_loss'],
                          Loss_summary['gan_loss'], Loss_summary['gen_loss'], Loss_summary['dis_loss'],
                          optimizer_G.param_groups[0]['lr'], (t1 - t0), (t3 - t2)))

            writer.add_scalar('fre_loss/hole_loss', Loss_summary['hole_loss'], iter)
            writer.add_scalar('fre_loss/valid_loss', Loss_summary['valid_loss'], iter)
            writer.add_scalar('fre_loss/bce_loss', Loss_summary['bce_loss'], iter)
            writer.add_scalar('fre_loss/sup_Loss', Loss_summary['sup_loss'], iter)
            writer.add_scalar('fre_loss/gan_loss', Loss_summary['gan_loss'], iter)
            writer.add_scalar('fre_loss/gen_loss', Loss_summary['gen_loss'], iter)
            writer.add_scalar('fre_loss/dis_loss', Loss_summary['dis_loss'], iter)
            for key, value in Loss_summary.items():
                Loss_summary[key] = 0

        if (iter+1) % (config['trainer']['save_freq']) == 0:
            if not config['model']['no_dis']:
                checkpoint(netG, netD, iter+1, optimizer_G, optimizer_D, scheduler_G, scheduler_D, config)
            else:
                checkpoint_nodis(netG, iter+1, optimizer_G, scheduler_G, config)

            PSNR, SSIM = test_model(netG, config, mode = 'online', iter=iter+1, writer=writer)
            if BestPSNR <= PSNR:
                BestPSNR = PSNR
                BestIter = iter+1
        torch.cuda.empty_cache()

    print('training is finished')
    print("Best Iter: {:0>2} \t Best PSNR: {:.3f}".format(BestIter, BestPSNR))
    writer.close()


def checkpoint(netG, netD, iter, optimizerG, optimizerD, schedulerG, schedulerD, config):
    save_model_path = 'experiments/' + config['name'] + '/weight'
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    model_name = 'gen_iter_{}.pth'.format(iter)
    torch.save(netG.state_dict(), os.path.join(save_model_path, model_name))
    model_name = 'dis_iter_{}.pth'.format(iter)
    torch.save(netD.state_dict(), os.path.join(save_model_path, model_name))

    save_state_path = 'experiments/' + config['name'] + '/state'
    if not os.path.exists(save_state_path):
        os.makedirs(save_state_path)
    opt_name = 'opt_{}.pth'.format(iter)
    if schedulerG:
        torch.save({'iteration': iter,
                    'optimG': optimizerG.state_dict(),
                    'scheG': schedulerG.state_dict(),
                    'optimD': optimizerD.state_dict(),
                    'scheD': schedulerD.state_dict(),
                    }
                   , os.path.join(save_state_path, opt_name))
    else:
        torch.save({'iteration': iter,
                    'optimG': optimizerG.state_dict(),
                    'optimD': optimizerD.state_dict(),
                    'scheD': schedulerD.state_dict(),
                    }
                   , os.path.join(save_state_path, opt_name))
    print('Checkpoint saved to {}'.format(os.path.join(save_model_path, model_name)))


def checkpoint_nodis(netG, iter, optimizerG, schedulerG, config):
    save_model_path = 'experiments/' + config['name'] + '/weight'
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    model_name = 'gen_iter_{}.pth'.format(iter)
    torch.save(netG.state_dict(), os.path.join(save_model_path, model_name))

    save_state_path = 'experiments/' + config['name'] + '/state'
    if not os.path.exists(save_state_path):
        os.makedirs(save_state_path)
    opt_name = 'opt_{}.pth'.format(iter)
    torch.save({'iteration': iter,
                'optimG': optimizerG.state_dict(),
                'scheG': schedulerG.state_dict(),
                }
               , os.path.join(save_state_path, opt_name))
    print('Checkpoint saved to {}'.format(os.path.join(save_model_path, model_name)))


if __name__ == '__main__':
    main()
