import os
import random
import sys
import shutil
from matplotlib import pyplot as plt
import numpy as np
import torch
import yaml
import seaborn as sns


def get_exp_name(args):
    train_env = ""
    for env in args.training_env:
        train_env += str(env)
        train_env += ","
    train_env = train_env[:-1]

    if len(args.training_class_env) == 0:
        class_env = "0.5"
    elif len(args.training_class_env) == 1:
        class_env = str(args.training_class_env[0])
    else:
        class_env = ""
        for env in args.training_class_env:
            class_env += str(env)
            class_env += ","
        class_env = class_env[:-1]
    if args.trainer == "ERM" or args.trainer == "GRAY":
        exp_name = "{}-{}-{}-TrEnv{}-PEnv{}-Epoch{}-wd{:.4f}-bs{}".format(
            args.trainer,
            args.arch,
            args.dataset,
            train_env,
            class_env,
            args.epochs,
            args.wd,
            args.batch_size
        )
    elif args.trainer == "IRM" or args.trainer == "IRMv0":
        exp_name = "{}-{}-{}-TrEnv{}-PEnv{}-Epoch{}-WarmEpoch{}-penalty{:.2f}-wd{:.4f}-bs{}".format(
            args.trainer,
            args.arch,
            args.dataset,
            train_env,
            class_env,
            args.epochs,
            args.warm_start,
            args.penalty_weight,
            args.wd,
            args.batch_size
        )
    else:
        exp_name = "{}-{}-{}-TrEnv{}-PEnv{}-Epoch{}-WarmEpoch{}-penalty{:.4f}-wd{:.4f}-bs{}".format(
            args.trainer,
            args.arch,
            args.dataset,
            train_env,
            class_env,
            args.epochs,
            args.warm_start,
            args.penalty_weight,
            args.wd,
            args.batch_size
        )

    return exp_name


def save_checkpoint(
        state, is_best, result_dir, filename="checkpoint.pth.tar"
):
    os.makedirs(result_dir, exist_ok=True)
    torch.save(state, os.path.join(result_dir, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(result_dir, filename),
            os.path.join(result_dir, "model_best.pth.tar"),
        )


def create_subdirs(sub_dir):
    os.mkdir(sub_dir)
    os.mkdir(os.path.join(sub_dir, "checkpoint"))


def trim_preceding_hyphens(st):
    i = 0
    while st[i] == "-":
        i += 1

    return st[i:]


def arg_to_varname(st: str):
    st = trim_preceding_hyphens(st)
    st = st.replace("-", "_")

    return st.split("=")[0]


def argv_to_vars(argv):
    var_names = []
    for arg in argv:
        if arg.startswith("-") and arg_to_varname(arg) != "config":
            var_names.append(arg_to_varname(arg))

    return var_names


def parse_configs_file(args):
    if args.configs is None:
        return
    # get commands from command line
    override_args = argv_to_vars(sys.argv)

    # load yaml file
    yaml_txt = open(args.configs).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print(f"=> Reading YAML config from {args.configs}")
    args.__dict__.update(loaded_yaml)


def setup_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

    def write_to_tensorboard(self, writer, prefix, global_step):
        for meter in self.meters:
            writer.add_scalar(f"{prefix}/{meter.name}", meter.val, global_step)


def get_data_per_epoch(data_per_epoch, train_stat, test_stat):
    assert len(data_per_epoch) > 6
    data_per_epoch[0].append(train_stat[0])  # Training acc
    data_per_epoch[1].append(test_stat[-1])  # Test avg acc
    data_per_epoch[2].append(test_stat[0] - test_stat[1])  # Test diff acc

    data_per_epoch[3].append(train_stat[1])  # Training loss
    data_per_epoch[4].append(train_stat[2])  # V0 loss
    data_per_epoch[5].append(train_stat[3])  # V1 loss
    data_per_epoch[6].append(train_stat[4])  # Stationary loss
    data_per_epoch[7].append(train_stat[5])  # Weight decay loss


def plot_trajectory(data_per_epoch, file_name):
    # Plotting
    plt.figure(figsize=(30, 50))
    sns.set_theme()
    plt.grid(visible=True, which='major', linestyle='-', linewidth=4)
    plt.grid(visible=True, which='minor')
    plt.minorticks_on()
    plt.rcParams['font.sans-serif'] = 'Times New Roman'

    fig, ax1 = plt.subplots()
    line_width = 2
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.set_yscale('log')
    l1 = ax1.plot(data_per_epoch[3], color="tab:orange", label="Train Loss", linewidth=line_width)
    l2 = ax1.plot(data_per_epoch[4], color="tab:brown", label="V0 Loss", linewidth=line_width)
    l3 = ax1.plot(data_per_epoch[5], color="tab:gray", label="V1 Loss", linewidth=line_width)
    l4 = ax1.plot(data_per_epoch[6], color="tab:red", label="Stationary Loss", linewidth=line_width)
    l5 = ax1.plot(data_per_epoch[7], color="tab:pink", label="WD Loss", linewidth=line_width)
    ax1.tick_params(axis='y', labelcolor="black")

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    line_width = 3
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)  # we already handled the x-label with ax1
    l6 = ax2.plot(np.array(data_per_epoch[0]) * 100, color="tab:green", label="Training Acc", linestyle="--", linewidth=line_width)
    l7 = ax2.plot(np.array(data_per_epoch[1]) * 100, color="tab:blue", label="Test Avg Acc", linestyle="--", linewidth=line_width)
    l8 = ax2.plot(np.array(data_per_epoch[2]) * 100, color="tab:purple", label="Test Acc Diffs", linestyle="--", linewidth=line_width)

    ax2.tick_params(axis='y', labelcolor="black")

    # Legends
    lg = l6 + l7 + l8 + l1 + l2 + l3 + l4 + l5
    labs = [l.get_label() for l in lg]
    ax1.legend(lg, labs, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.title("IRM Training Dynamics")
    # name = "_w=1_real"

    if isinstance(file_name, list):
        for file in file_name:
            plt.savefig(file)
    else:
        plt.savefig(file_name)
    # plt.show()
    plt.clf()
    plt.cla()
    plt.close('all')
