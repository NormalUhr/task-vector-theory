import time
import torch

from utils.general_utils import AverageMeter, ProgressMeter
from utils.training_utils import criterion, penalty_v1, penalty_stationary, mean_accuracy


def train(
        model, args, device, train_loaders, optimizer, lr_scheduler, epoch
):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    penalty_v1_losses = AverageMeter("Penalty v1", ":.4f")
    penalty_stationary_loss = AverageMeter("Stationary Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")

    batch_total = torch.sum(torch.tensor([len(loader) for loader in train_loaders])).item()
    progress = ProgressMeter(
        batch_total,
        [batch_time, losses, top1, penalty_v1_losses, penalty_stationary_loss],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    end = time.time()

    args.penalty_weight = 0.0
    args.warm_start = 0

    data_iters = [iter(loader) for loader in train_loaders]

    stop_flag = False

    total_num = 0

    iter_num = 0

    while not stop_flag:
        for data_iter in data_iters:
            iter_num += 1
            try:
                images, labels = data_iter.next()
            except StopIteration:
                stop_flag = True
                break

            total_num += images.shape[0]

            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            penalty_v1_loss = penalty_v1(logits, labels)
            penalty_stationary(model, images, labels, 0)
            acc1 = mean_accuracy(logits, labels)

            losses.update(loss.item(), images.shape[0])
            top1.update(acc1.item(), images.shape[0])
            penalty_v1_losses.update(penalty_v1_loss.item(), images.shape[0])

            weight_norm = torch.tensor(0.).to(device)
            for w in model.parameters():
                weight_norm += w.norm().pow(2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if iter_num % args.print_freq == 0:
                progress.display(iter_num)



