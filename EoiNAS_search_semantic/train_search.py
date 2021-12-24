import os
import sys
import time
import glob
import numpy as np
import torch
import math
import utils_original
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import copy
from model_search import Network
from genotypes import PRIMITIVES
from genotypes import Genotype
from dataloaders.datasets import cityscapes
from torch.utils.data import DataLoader
from utils.loss import SegmentationLosses
from utils.metrics import Evaluator


parser = argparse.ArgumentParser("cityscapes")
parser.add_argument('--workers', type=int, default=4, help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=2, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=90, help='num of training epochs')
parser.add_argument('--layers', type=int, default=10, help='total number of layers')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='/tmp/checkpoints/', help='experiment path')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--arch_learning_rate', type=float, default=0.003, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--temperature_init', type=float, default=5.0, help='temperature for arch encoding')
parser.add_argument('--temperature_decay', type=float, default=0.965, help='temperature_decay for arch encoding')
parser.add_argument('--tmp_data_dir', type=str, default='/tmp/cache/', help='temp data dir')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout rate of skip connect')
parser.add_argument('--out_stride', type=int, default=16, help='network output stride (default: 8)')
parser.add_argument('--dataset', type=str, default='cityscapes', choices=['pascal', 'coco', 'cityscapes'],help='dataset name (default: pascal)')
parser.add_argument('--crop_size', type=int, default=320, help='crop image size')
parser.add_argument('--resize', type=int, default=512, help='resize image size')
parser.add_argument('--loss_type', type=str, default='ce', choices=['ce', 'focal'], help='loss func type (default: ce)')
parser.add_argument('--no_cuda', action='store_true', default = False, help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.save = '{}-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))
utils_original.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)

    #  prepare dataset
    train_set1, train_set2 = cityscapes.sp(args, split='train')
    val_set = cityscapes.CityscapesSegmentation(args, split='val')
    num_class = train_set1.NUM_CLASSES
    train_loader1 = DataLoader(train_set1, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
    train_loader2 = DataLoader(train_set2, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

    criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)
    
    # build Network
    switches = []
    for i in range(14):
        switches.append([True for j in range(len(PRIMITIVES))])
    switches_normal = copy.deepcopy(switches)
    switches_reduce = copy.deepcopy(switches)
    switches_normal_2 = []
    switches_reduce_2 = []

    # To be moved to args
    eps_no_archs = 20
    model = Network(num_class, args.layers , criterion, crop_size=args.crop_size, switches_normal = switches_normal, switches_reduce = switches_reduce, p = args.dropout_rate)
    model = nn.DataParallel(model)
    model = model.cuda()
    logging.info("param size = %fMB", utils_original.count_parameters_in_MB(model))
    network_params = []
    for k, v in model.named_parameters():
        if not (k.endswith('alphas_normal') or k.endswith('alphas_reduce')):
            network_params.append(v)
    optimizer = torch.optim.SGD(
            network_params,
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
    optimizer_a = torch.optim.Adam(model.module.arch_parameters(),
                lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
    # Define lr scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    # Define Evaluator
    evaluator = Evaluator(num_class)
    epochs = args.epochs
    scale_factor = 0.2

    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('Epoch: %d lr: %e', epoch, lr)
        epoch_start = time.time()
        # training
        if epoch < eps_no_archs:
            model.module.p = float(args.dropout_rate) * (epochs - epoch - 1) / epochs
            model.module.update_p()
            temperature = args.temperature_init
            train_acc, train_acc_class, train_mIoU, train_FWIoU, train_obj = train(train_loader1, train_loader2, model, network_params, temperature, criterion, optimizer, optimizer_a, evaluator, train_arch=False)

        else:
            model.module.p = float(args.dropout_rate) * np.exp(-(epoch - eps_no_archs) * scale_factor)
            model.module.update_p()
            temperature = args.temperature_init * math.pow(args.temperature_decay, (epoch - eps_no_archs))
            train_acc, train_acc_class, train_mIoU, train_FWIoU, train_obj = train(train_loader1, train_loader2, model, network_params, temperature, criterion, optimizer, optimizer_a, evaluator, train_arch=True)

        logging.info('Train_acc %f', train_acc)
        logging.info('Train_acc_class %f', train_acc_class)
        logging.info('Train_mIoU %f', train_mIoU)
        logging.info('Train_FWIoU %f', train_FWIoU)
        epoch_duration = time.time() - epoch_start
        logging.info('Epoch time: %ds', epoch_duration)

        # validation
        # valid_acc, valid_mIoU, valid_obj = infer(val_loader, model, criterion, evaluator)
        # logging.info('valid_acc %f', valid_acc)
        # logging.info('valid_mIoU %f', valid_mIoU)

        if epoch in range(29, 90, 10):
            print('------Dropping %d paths------' % 1)
            # Save switches info for s-c refinement.
            if epoch == epochs - 1:
                switches_normal_2 = copy.deepcopy(switches_normal)
                switches_reduce_2 = copy.deepcopy(switches_reduce)
            # drop operations with low architecture weights
            arch_param = model.module.arch_parameters()  
            normal_prob = F.softmax(arch_param[0], dim=-1).data.cpu().numpy()
            for i in range(14):
                idxs = []
                for j in range(len(PRIMITIVES)):
                    if switches_normal[i][j]:
                        idxs.append(j)
                if epoch == epochs - 1:
                    # for the last stage, drop all Zero operations
                    drop = get_min_k_no_zero(normal_prob[i, :], idxs, 1)
                else:
                    drop = get_min_k(normal_prob[i, :], 1)
                for idx in drop:
                    switches_normal[i][idxs[idx]] = False
            reduce_prob = F.softmax(arch_param[1], dim=-1).data.cpu().numpy()
            for i in range(14):
                idxs = []
                for j in range(len(PRIMITIVES)):
                    if switches_reduce[i][j]:
                        idxs.append(j)
                if epoch == epochs - 1:
                    drop = get_min_k_no_zero(reduce_prob[i, :], idxs, 1)
                else:
                    drop = get_min_k(reduce_prob[i, :], 1)
                for idx in drop:
                    switches_reduce[i][idxs[idx]] = False
            logging.info('switches_normal = %s', switches_normal)    
            logging_switches(switches_normal)
            logging.info('switches_reduce = %s', switches_reduce)
            logging_switches(switches_reduce)

            if epoch in range(29,80,10):
                model.module.switches_normal = switches_normal
                model.module.switches_reduce = switches_reduce
                indexes_normal = []
                indexes_reduce = []
                for i in range(14):
                    index = np.argmin(normal_prob[i])
                    indexes_normal.append(index)
                for i in range(14):
                    index = np.argmin(reduce_prob[i])
                    indexes_reduce.append(index)
                model.module.update_switches(indexes_normal, indexes_reduce)

                column_len = 0
                for j in range(len(PRIMITIVES)):
                    if switches_normal[1][j]:
                        column_len = column_len + 1
                arch_normal = model.module.arch_parameters()[0].data.cpu().numpy()
                arch_reduce = model.module.arch_parameters()[1].data.cpu().numpy()
                w_drop_normal = []
                w_drop_reduce = []
                for i in range(14):
                    w_drop = np.argmin(arch_normal[i])
                    w_drop_normal.append(w_drop)
                for i in range(14):
                    w_drop = np.argmin(arch_reduce[i])
                    w_drop_reduce.append(w_drop)
                model.module.update_arch_parameters(w_drop_normal, w_drop_reduce, column_len)
                optimizer_a = torch.optim.Adam(model.module.arch_parameters(),
                                               lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                               weight_decay=args.arch_weight_decay)

    utils_original.save(model, os.path.join(args.save, 'weights.pt'))
    arch_param = model.module.arch_parameters()
    normal_prob = F.softmax(arch_param[0], dim=-1).data.cpu().numpy()
    reduce_prob = F.softmax(arch_param[1], dim=-1).data.cpu().numpy()
    normal_final = [0 for idx in range(14)]
    reduce_final = [0 for idx in range(14)]
    # remove all Zero operations
    for i in range(14):
        if switches_normal_2[i][0] == True:
            normal_prob[i][0] = 0
        normal_final[i] = max(normal_prob[i])
        if switches_reduce_2[i][0] == True:
            reduce_prob[i][0] = 0
        reduce_final[i] = max(reduce_prob[i])
    # Generate Architecture, similar to DARTS
    keep_normal = [0, 1]
    keep_reduce = [0, 1]
    n = 3
    start = 2
    for i in range(3):
        end = start + n
        tbsn = normal_final[start:end]
        tbsr = reduce_final[start:end]
        edge_n = sorted(range(n), key=lambda x: tbsn[x])
        keep_normal.append(edge_n[-1] + start)
        keep_normal.append(edge_n[-2] + start)
        edge_r = sorted(range(n), key=lambda x: tbsr[x])
        keep_reduce.append(edge_r[-1] + start)
        keep_reduce.append(edge_r[-2] + start)
        start = end
        n = n + 1
    # set switches according the ranking of arch parameters
    for i in range(14):
        if not i in keep_normal:
            for j in range(len(PRIMITIVES)):
                switches_normal[i][j] = False
        if not i in keep_reduce:
            for j in range(len(PRIMITIVES)):
                switches_reduce[i][j] = False
    # translate switches into genotype
    genotype = parse_network(switches_normal, switches_reduce)
    logging.info(genotype)


def train(train_queue, valid_queue, model, network_params, temperature, criterion, optimizer, optimizer_a, evaluator, train_arch=True):
    objs = utils_original.AvgrageMeter()
    evaluator.reset()

    for step, sample in enumerate(train_queue):
        input, target = sample['image'], sample['label']
        model.train()
        n = input.size(0)
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        if train_arch:
            try:
                search = next(valid_queue_iter)
                input_search, target_search = search['image'], search['label']
            except:
                valid_queue_iter = iter(valid_queue)
                search = next(valid_queue_iter)
                input_search, target_search = search['image'], search['label']
            input_search = input_search.cuda()
            target_search = target_search.cuda(non_blocking=True)
            optimizer_a.zero_grad()
            logits_a = model(input_search, temperature)
            loss_a = criterion(logits_a, target_search)
            loss_a.backward()
            nn.utils.clip_grad_norm_(model.module.arch_parameters(), args.grad_clip)
            optimizer_a.step()

        optimizer.zero_grad()
        logits = model(input, temperature)
        loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(network_params, args.grad_clip)
        optimizer.step()

        pred = logits.data.cpu().numpy()
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        # Add batch sample into evaluator
        evaluator.add_batch(target, pred)

        objs.update(loss.data.item(), n)
        if step % args.report_freq == 0:
            logging.info('TRAIN Step: %03d Objs: %e', step, objs.avg)

    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

    return Acc, Acc_class, mIoU, FWIoU, objs.avg


def infer(valid_queue, model, criterion, evaluator):
    objs = utils_original.AvgrageMeter()
    evaluator.reset()
    model.eval()

    for step, sample in enumerate(valid_queue):
        input, target = sample['image'], sample['label']
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits = model(input)
            loss = criterion(logits, target)

        pred = logits.data.cpu().numpy()
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        # Add batch sample into evaluator
        evaluator.add_batch(target, pred)

        n = input.size(0)
        objs.update(loss.data.item(), n)
        if step % args.report_freq == 0:
            logging.info('TRAIN Step: %03d Objs: %e', step, objs.avg)

    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

    return Acc, Acc_class, mIoU, FWIoU, objs.avg


def parse_network(switches_normal, switches_reduce):

    def _parse_switches(switches):
        n = 2
        start = 0
        gene = []
        step = 4
        for i in range(step):
            end = start + n
            for j in range(start, end):
                for k in range(len(switches[j])):
                    if switches[j][k]:
                        gene.append((PRIMITIVES[k], j - start))
            start = end
            n = n + 1
        return gene
    gene_normal = _parse_switches(switches_normal)
    gene_reduce = _parse_switches(switches_reduce)
    
    concat = range(2, 6)
    
    genotype = Genotype(
        normal=gene_normal, normal_concat=concat, 
        reduce=gene_reduce, reduce_concat=concat
    )
    
    return genotype

def get_min_k(input_in, k):
    input = copy.deepcopy(input_in)
    index = []
    for i in range(k):
        idx = np.argmin(input)
        index.append(idx)
        input[idx] = 1
    
    return index
def get_min_k_no_zero(w_in, idxs, k):
    w = copy.deepcopy(w_in)
    index = []
    if 0 in idxs:
        zf = True 
    else:
        zf = False
    if zf:
        w = w[1:]
        index.append(0)
        k = k - 1
    for i in range(k):
        idx = np.argmin(w)
        w[idx] = 1
        if zf:
            idx = idx + 1
        index.append(idx)
    return index
        
def logging_switches(switches):
    for i in range(len(switches)):
        ops = []
        for j in range(len(switches[i])):
            if switches[i][j]:
                ops.append(PRIMITIVES[j])
        logging.info(ops)


if __name__ == '__main__':
    start_time = time.time()
    main() 
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Total searching time: %ds', duration)
