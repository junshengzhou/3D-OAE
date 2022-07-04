import torch
import torch.nn as nn
from datasets.ShapeNetPart import PartNormalDataset
from models.Point_OAE_seg import get_loss
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter

import numpy as np
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
from torchvision import transforms
from torch.utils.data import DataLoader

from utils.seg_utils import compute_overall_iou, to_categorical

class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        elif type(acc).__name__ == 'Acc_Metric':
            self.acc = acc.acc
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    # =========== Dataloader =================
    train_data = PartNormalDataset(npoints=2048, split='trainval', normalize=False)
    print("The number of training data is:%d", len(train_data))

    test_data = PartNormalDataset(npoints=2048, split='test', normalize=False)
    print("The number of test data is:%d", len(test_data))

    train_dataloader = DataLoader(train_data, batch_size=config.total_bs, shuffle=True, num_workers=args.num_workers,
                              drop_last=True)

    test_dataloader = DataLoader(test_data, batch_size=config.total_bs, shuffle=False, num_workers=args.num_workers,
                             drop_last=False)

    # build model
    base_model = builder.model_builder(config.model)
    
    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    best_metrics_vote = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Acc_Metric(best_metrics)
    else:
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('Training from scratch', logger = logger)

    if args.use_gpu:    
        base_model.to(args.local_rank)
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['loss', 'acc'])
        num_iter = 0
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)

        npoints = config.npoints
        for idx, (points, label, target, _) in enumerate(train_dataloader):
            B, N, _ = points.shape
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)
            
            points = points.cuda()  # B, N, 3
            label = label.long().cuda()    # B, 1
            target = target.long().cuda()  # B, N
            label = to_categorical(label.squeeze(1), 16)    # num_classes=16   B, num_class

            ret, _ = base_model(points, label)

            loss = base_model.module.get_loss_seg(ret, target)
            batch_iou = compute_overall_iou(ret, target, 50)    # num_part=50
            batch_iou = ret.new_tensor([np.sum(batch_iou)], dtype=torch.float64)/B  # same device with seg_pred!!!

            _loss = loss

            _loss.backward()

            # forward
            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            losses.update([loss.item(), batch_iou.item()])

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/TrainIou', batch_iou.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)


            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 10 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss+Iou = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Loss_2', losses.avg(1), epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger, len_data=len(test_data))

            better = metrics.better_than(best_metrics)
            # Save ckeckpoints
            if better:
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)

        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)      
        if (config.max_epoch - epoch) < 10:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)     
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

def validate(base_model, test_dataloader, epoch, val_writer, args, config, logger = None, len_data=1):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints
    total_iou = 0.0
    with torch.no_grad():
        for idx, (points, label, target, _) in enumerate(test_dataloader):
            points = points.cuda()  # B, N, 3
            label = label.long().cuda()    # B, 1
            target = target.long().cuda()  # B, N
            label = to_categorical(label.squeeze(1), 16)    # num_classes=16   B, num_class

            # points = test_transforms(points)

            ret, _ = base_model(points, label)

            batch_iou = compute_overall_iou(ret, target, 50)    # num_part=50
            batch_iou = ret.new_tensor([np.sum(batch_iou)], dtype=torch.float64)  # same device with seg_pred!!!
            total_iou += batch_iou

        mean_iou = total_iou/len_data
        print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch, mean_iou), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/Iou', mean_iou, epoch)

    return Acc_Metric(mean_iou)


def test_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    # =========== Dataloader =================
    test_data = PartNormalDataset(npoints=2048, split='test', normalize=False)
    print("The number of test data is:%d", len(test_data))

    test_dataloader = DataLoader(test_data, batch_size=config.total_bs, shuffle=False, num_workers=args.num_workers,
                             drop_last=False)

    # build model
    base_model = builder.model_builder(config.model)
    
    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    best_metrics_vote = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Acc_Metric(best_metrics)
    else:
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('Training from scratch', logger = logger)

    if args.use_gpu:    
        base_model.to(args.local_rank)
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    # trainval
    # training
    metrics = validate_perclass(base_model, test_dataloader, 0, val_writer, args, config, logger=logger, len_data=len(test_data))

def validate_perclass(base_model, test_dataloader, epoch, val_writer, args, config, logger = None, len_data=1):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []

    pred_shapes = []
    pred_labels = []
    gt_labels = []

    final_total_per_cat_iou = np.zeros(16).astype(np.float32)
    final_total_per_cat_seen = np.zeros(16).astype(np.int32)
    npoints = config.npoints
    total_iou = 0.0
    with torch.no_grad():
        for idx, (points, label, target, _) in enumerate(test_dataloader):
            points = points.cuda()  # B, N, 3
            label = label.long().cuda()    # B, 1
            target = target.long().cuda()  # B, N
            label_input = to_categorical(label.squeeze(1), 16)    # num_classes=16   B, num_class

            ret, _ = base_model(points, label_input)

            pred_shapes.append(points)
            pred_labels.append(ret.max(dim=2)[1])
            gt_labels.append(target)

            batch_iou = compute_overall_iou(ret, target, 50)    # num_part=50

            for shape_idx in range(ret.size(0)):  # sample_idx
                cur_gt_label = label[shape_idx]  # label[sample_idx], denotes current sample belongs to which cat
                final_total_per_cat_iou[cur_gt_label] += batch_iou[shape_idx]  # add the iou belongs to this cat
                final_total_per_cat_seen[cur_gt_label] += 1  # count the number of this cat is chosen
                
            batch_iou = ret.new_tensor([np.sum(batch_iou)], dtype=torch.float64)  # same device with seg_pred!!!
            total_iou += batch_iou

        mean_iou = total_iou/len_data

        pred_shapes = torch.cat(pred_shapes, dim=0).reshape(-1, 2048, 3)
        pred_labels = torch.cat(pred_labels, dim=0).reshape(-1, 2048)
        gt_labels = torch.cat(gt_labels, dim=0).reshape(-1, 2048)
        np.save('./visuals_seg/shapes.npy', pred_shapes.detach().cpu().numpy())
        np.save('./visuals_seg/gts.npy', gt_labels.detach().cpu().numpy())
        np.save('./visuals_seg/preds.npy', pred_labels.detach().cpu().numpy())

        for cat_idx in range(16):
            if final_total_per_cat_seen[cat_idx] > 0:  # indicating this cat is included during previous iou appending
                final_total_per_cat_iou[cat_idx] = final_total_per_cat_iou[cat_idx] / final_total_per_cat_seen[cat_idx]  # avg class iou across all samples
                print_log('class%d iou ='%(cat_idx) + str(final_total_per_cat_iou[cat_idx]))

        print_log('[Validation] EPOCH: %d  miouI = %.4f miouC %.5f' % (epoch, mean_iou, np.mean(final_total_per_cat_iou)), logger=logger)


        if args.distributed:
            torch.cuda.synchronize()
    

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/Iou', mean_iou, epoch)

    return Acc_Metric(mean_iou)
