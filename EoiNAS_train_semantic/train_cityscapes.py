import os
import sys
import time
import pdb
import glob
import warnings
import numpy as np
import random
import logging
import argparse
import utils_original
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.optim as optim
import genotypes
from dataloaders.datasets import cityscapes
import torch.utils
from utils.loss import build_criterion
from utils.step_lr_scheduler import Iter_LR_Scheduler
from utils.metrics import Evaluator
from torch.autograd import Variable
from train_model.model import NetworkCityscapes


parser = argparse.ArgumentParser("Cityscapes")
parser.add_argument('--workers', type=int, default=4, help='number of workers to load dataset')
parser.add_argument('--data', type=str, default='../data/imagenet/', help='location of the data corpus')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--layers', type=int, default=10, help='total number of layers')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--train_Coarse_checkpoint', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='GNAS', help='which architecture to use')
parser.add_argument('--parallel', action='store_true', default=False, help='data parallelism')
parser.add_argument('--gpu_str', type=str, default='0', help='test time gpu device id')
parser.add_argument('--epochs', type=int, default=300, help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=3, help='batch size')
parser.add_argument('--base_lr', type=float, default=0.005, help='base learning rate')
parser.add_argument('--warmup_start_lr', type=float, default=5e-6, help='warm up learning rate')
parser.add_argument('--lr-step', type=float, default=None)
parser.add_argument('--warmup-iters', type=int, default=1000)
parser.add_argument('--min-lr', type=float, default=None)
parser.add_argument('--crop_size', type=int, default=768, help='image crop size')
parser.add_argument('--eval_scales', type=int, default=[1.0], help='eval_scales')
parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')
parser.add_argument('--criterion', default='Ohem', type=str)
parser.add_argument('--mode', default='poly', type=str, help='how lr decline')
args = parser.parse_args()

args.save = '{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils_original.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    train_set = cityscapes.CityscapesSegmentation(args, split='retrain')
    num_classes = train_set.NUM_CLASSES
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers, drop_last=True)
    val_set = cityscapes.CityscapesSegmentation(args, split='reval')
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers, drop_last=True)

    genotype = eval("genotypes.%s" % args.arch)
    print('---------Genotype---------')
    logging.info(genotype)
    print('--------------------------')
    model = NetworkCityscapes(args.init_channels, num_classes, args.layers, args.crop_size, genotype)
    if args.parallel:
        model = nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

#    utils_original.load(model, args.train_Coarse_checkpoint)  
            
    # checkpoint = torch.load(args.train_Coarse_checkpoint)
    # model.load_state_dict(checkpoint['state_dict'])
            

    logging.info("param size = %fMB", utils_original.count_parameters_in_MB(model))

    if args.criterion == 'Ohem':
        args.thresh = 0.7
        args.crop_size = [args.crop_size, args.crop_size] if isinstance(args.crop_size, int) else args.crop_size
        args.n_min = int((args.batch_size / len(args.gpu_str) * args.crop_size[0] * args.crop_size[1]) // 16)
    criterion = build_criterion(args)

    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=5e-4)
    max_iteration = len(train_loader) * args.epochs
    scheduler = Iter_LR_Scheduler(args, max_iteration, len(train_loader))
    evaluator = Evaluator(num_classes)
    start_epoch = 0
    best_mIoU = 0.0

    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {0}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_mIoU = checkpoint['best_mIoU']
            print('=> loaded checkpoint {0} (epoch {1})'.format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError('=> no checkpoint found at {0}'.format(args.resume))

   
    for epoch in range(start_epoch, args.epochs):
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        train_obj = train(train_loader, model, criterion, optimizer, scheduler, epoch)
        logging.info('Train_loss: %f', train_obj)

        # validation
        valid_acc, valid_acc_class, valid_mIoU, valid_FWIoU = valid(val_loader, model, evaluator, num_classes)
        logging.info('Valid_acc %f', valid_acc)
        logging.info('Valid_mIoU %f', valid_mIoU)
        logging.info('Valid_acc_class %f', valid_acc_class)
        logging.info('Valid_FWIoU %f', valid_FWIoU)

        is_best = False
        if valid_mIoU > best_mIoU:
            best_mIoU = valid_mIoU
            is_best = True

        utils_original.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mIoU': best_mIoU,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.save)

        logging.info('best_mIoU %f', best_mIoU)

def train(train_queue, model, criterion, optimizer, scheduler, cur_epoch):
    model.train()
    objs = utils_original.AvgrageMeter()

    for i, sample in enumerate(train_queue):
        cur_iter = cur_epoch * len(train_queue) + i
        scheduler(optimizer, cur_iter)
        inputs = sample['image'].cuda()
        target = sample['label'].cuda()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        if np.isnan(loss.item()) or np.isinf(loss.item()):
            pdb.set_trace()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        objs.update(loss.data.item(), args.batch_size)

        if i % args.report_freq == 0:
            logging.info('TRAIN Step: %03d Objs: %e lr: %f', i, objs.avg, scheduler.get_lr(optimizer))

    return objs.avg


def valid(valid_queue, model, evaluator, num_classes):
    evaluator.reset()
    model.eval()

    for i, sample in enumerate(valid_queue):
        inputs, target = sample['image'], sample['label']
        N, H, W = target.shape
        total_outputs = torch.zeros((N, num_classes, H, W)).cuda()
        with torch.no_grad():
            for j, scale in enumerate(args.eval_scales):
                new_scale = [int(H * scale), int(W * scale)]
                inputs = F.interpolate(inputs, new_scale, None, 'bilinear', True)
                inputs = inputs.cuda()
                outputs = model(inputs)
                outputs = F.interpolate(outputs, (H, W), None, 'bilinear', True)
                total_outputs += outputs

            pred = total_outputs.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            evaluator.add_batch(target, pred)

            if i % args.report_freq == 0:
                logging.info('VALID Step: %03d', i)

    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

    return Acc, Acc_class, mIoU, FWIoU


if __name__ == "__main__":
    main()



