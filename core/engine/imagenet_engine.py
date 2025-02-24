import os
import time
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from core import models
from core.datasets.data_zoo import get_data_by_name
from core.utils.lr_scheduler import get_scheduler_by_name
from core.utils.copy_weights import copy_weights
from core.utils.resume import resume_from_checkpoint
from core.engine.base import validate, train, save_checkpoint

best_acc1 = 0


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        try:
            model = models.__dict__[args.arch](pretrained=True, num_classes=args.classes, args=args)
        except TypeError:
            print(F"Parameter Type Error, fixing...")
            model = models.__dict__[args.arch](pretrained=True, num_classes=args.classes)

    else:
        print("=> creating model '{}'".format(args.arch))
        # model = models.__dict__[args.arch](num_classes=args.classes, args=args)
        try:
            model = models.__dict__[args.arch](num_classes=args.classes, args=args)
        except TypeError:
            print(F"Parameter Type Error, fixing...")
            model = models.__dict__[args.arch](num_classes=args.classes)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    # criterion = nn.MultiLabelSoftMarginLoss().cuda(args.gpu)

    adjust_learning_rate = get_scheduler_by_name(args.lr_scheduler)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), args.lr,
    #                              weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        args, model, optimizer, best_acc1 = resume_from_checkpoint(args, model, optimizer, best_acc1)

    cudnn.benchmark = True

    # Data loading code
    train_dataset, val_dataset, test_dataset = get_data_by_name(args.data_format, data_dir=args.data)

    # train data loader is here, distribute is support #
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        # val_sampler = None
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ #

    # val data loader is here #
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)
    # ^^^^^^^^^^^^^^^^^^^^^^^ #
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    if args.test:
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        # validate(train_loader, model, criterion, args)
        # print('TEST IN TRAIN SET')
        # validate(val_loader, model, criterion, args)
        # print('TEST IN VAL SET')
        validate(test_loader, model, criterion, args)
        print('TEST IN TEST SET')
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch, args)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        start = time.time()
        train(train_loader, model, criterion, optimizer, epoch, args)
        print(f"epoch time {time.time() - start}")
        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best)
            if is_best:
                copy_weights(args, epoch)

    copy_weights(args, args.epochs)