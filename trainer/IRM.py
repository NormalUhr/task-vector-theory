import time
import torch

from utils.general_utils import AverageMeter, ProgressMeter
from utils.training_utils import criterion, penalty_v1, mean_accuracy, penalty_stationary, penalty_v0


def train(
        model, args, device, train_loaders, optimizer, scheduler, epoch
):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    penalty_v1_losses = AverageMeter("Lv1", ":.4f")
    penalty_v0_losses = AverageMeter("Lv0", ":.4f")
    penalty_stationary_losses = AverageMeter("LS", ":.4f")
    reg_losses = AverageMeter("Reg", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")

    batch_total = torch.sum(torch.tensor([len(loader) for loader in train_loaders])).item()
    progress = ProgressMeter(
        batch_total,
        [batch_time, top1, losses, penalty_v0_losses, penalty_v1_losses, penalty_stationary_losses],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    end = time.time()

    data_iters = [iter(loader) for loader in train_loaders]

    stop_flag = False

    iter_num = 0

    penalty_weight = (args.penalty_weight
                      if epoch >= args.warm_start else 1.0)

    while 1:
        envs = [{} for _ in data_iters]
        iter_num += 1
        batch_num = 0
        for env_num, data_iter in enumerate(data_iters):
            try:
                images, labels = data_iter.next()
            except StopIteration:
                stop_flag = True
                break

            batch_num += images.shape[0]

            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            envs[env_num]["loss"] = loss
            envs[env_num]["penalty_v1"] = penalty_v1(logits, labels)
            envs[env_num]["penalty_v0"] = penalty_v0(model, images, labels)
            envs[env_num]["penalty_stationary"] = penalty_stationary(model, images, labels, 0)
            envs[env_num]["acc"] = mean_accuracy(logits, labels)

        if stop_flag:
            break

        training_loss = torch.stack([env["loss"] for env in envs]).mean()
        training_acc = torch.stack([env["acc"] for env in envs]).mean()
        penalty_v1_loss = torch.stack([env["penalty_v1"] for env in envs]).mean()
        penalty_v0_loss = torch.stack([env["penalty_v0"] for env in envs]).mean()
        penalty_stationary_loss = torch.stack([env["penalty_stationary"] for env in envs]).mean()

        loss = torch.tensor(0.0).to(device)

        # Training loss
        loss += training_loss

        # Weight Decay
        weight_norm = torch.tensor(0.).to(device)
        for w in model.parameters():
            weight_norm += w.norm().pow(2)
        loss += args.wd * weight_norm

        # Invariance Penalty
        loss += penalty_weight * penalty_v1_loss
        if penalty_weight > 1.0:
            loss /= penalty_weight

        # Model Parameter Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Update Statistics
        losses.update(training_loss.item(), batch_num)
        top1.update(training_acc.item(), batch_num)
        penalty_v1_losses.update(penalty_v1_loss.item() * penalty_weight, batch_num)
        penalty_v0_losses.update(penalty_v0_loss.item() * penalty_weight, batch_num)
        penalty_stationary_losses.update(penalty_stationary_loss.item() * penalty_weight, batch_num)
        reg_losses.update(weight_norm.item(), batch_num)
        batch_time.update(time.time() - end)
        end = time.time()

        # Display
        if iter_num % args.print_freq == 0:
            progress.display(iter_num)

    return [top1.avg, losses.avg, penalty_v0_losses.avg, penalty_v1_losses.avg, penalty_stationary_losses.avg, reg_losses.avg]



