import math
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader

DOWNLOAD_DIR = "../data"


def extract_tensors_from_loader(dl, repeat=1, transform_fn=None):
    X, Y = [], []
    for _ in range(repeat):
        for xb, yb in dl:
            if transform_fn:
                xb, yb = transform_fn(xb, yb)
            X.append(xb)
            Y.append(yb)
    X = torch.FloatTensor(torch.cat(X))
    Y = torch.LongTensor(torch.cat(Y))
    return X, Y


def extract_numpy_from_loader(dl, repeat=1, transform_fn=None):
    X, Y = extract_tensors_from_loader(dl, repeat=repeat, transform_fn=transform_fn)
    return X.numpy(), Y.numpy()


def get_mnist(fpath=DOWNLOAD_DIR, flatten=False, binarize=False, normalize=False, y0={0, 1, 2, 3, 4}):
    """get preprocessed mnist torch.TensorDataset class"""

    def _to_torch(d):
        X, Y = [], []
        for xb, yb in d:
            X.append(xb)
            Y.append(yb)
        return torch.Tensor(np.stack(X)), torch.LongTensor(np.stack(Y))

    to_tensor = torchvision.transforms.ToTensor()
    to_flat = torchvision.transforms.Lambda(lambda X: X.reshape(-1).squeeze())
    to_norm = torchvision.transforms.Normalize((0.5,), (0.5,))
    to_binary = torchvision.transforms.Lambda(lambda y: 0 if y in y0 else 1)

    transforms = [to_tensor]
    if normalize: transforms.append(to_norm)
    if flatten: transforms.append(to_flat)
    tf = torchvision.transforms.Compose(transforms)
    ttf = to_binary if binarize else None

    X_tr = torchvision.datasets.MNIST(fpath, download=True, transform=tf, target_transform=ttf)
    X_te = torchvision.datasets.MNIST(fpath, download=True, train=False, transform=tf, target_transform=ttf)

    return _to_torch(X_tr), _to_torch(X_te)


def get_cifar(fpath=DOWNLOAD_DIR, use_cifar10=False, flatten_data=False, transform_type='none',
              means=None, std=None, use_grayscale=False, binarize=False, normalize=False, y0={0, 1, 2, 3, 4}):
    """get preprocessed cifar torch.Dataset class"""

    if transform_type == 'none':
        normalize_cifar = lambda: torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        tensorize = torchvision.transforms.ToTensor()
        to_grayscale = torchvision.transforms.Grayscale()
        flatten = torchvision.transforms.Lambda(lambda X: X.reshape(-1).squeeze())

        transforms = [tensorize]
        if use_grayscale: transforms = [to_grayscale] + transforms
        if normalize: transforms.append(normalize_cifar())
        if flatten_data: transforms.append(flatten)
        tr_transforms = te_transforms = torchvision.transforms.Compose(transforms)

    if transform_type == 'basic':
        normalize_cifar = lambda: torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

        tr_transforms = [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor()
        ]

        te_transforms = [
            torchvision.transforms.Resize(32),
            torchvision.transforms.CenterCrop(32),
            torchvision.transforms.ToTensor(),
        ]

        if normalize:
            tr_transforms.append(normalize_cifar())
            te_transforms.append(normalize_cifar())

        tr_transforms = torchvision.transforms.Compose(tr_transforms)
        te_transforms = torchvision.transforms.Compose(te_transforms)

    to_binary = torchvision.transforms.Lambda(lambda y: 0 if y in y0 else 1)
    target_transforms = to_binary if binarize else None
    dset = 'cifar10' if use_cifar10 else 'cifar100'
    func = torchvision.datasets.CIFAR10 if use_cifar10 else torchvision.datasets.CIFAR100

    X_tr = func(fpath, download=True, transform=tr_transforms, target_transform=target_transforms)
    X_te = func(fpath, download=True, train=False, transform=te_transforms, target_transform=target_transforms)

    return X_tr, X_te


def _to_dl(X, Y, bs, shuffle=True):
    return DataLoader(TensorDataset(torch.Tensor(X), torch.LongTensor(Y)), batch_size=bs, shuffle=shuffle)


def get_binary_datasets(X, Y, y1, y2, image_width=28, use_cnn=False):
    assert type(X) is np.ndarray and type(Y) is np.ndarray
    idx0 = (Y == y1).nonzero()[0]
    idx1 = (Y == y2).nonzero()[0]
    idx = np.concatenate((idx0, idx1))
    X_, Y_ = X[idx, :], (Y[idx] == y2).astype(int)
    P = np.random.permutation(len(X_))
    X_, Y_ = X_[P, :], Y_[P]
    if use_cnn: X_ = X_.reshape(X.shape[0], -1, image_width)[:, None, :, :]
    return X_[P, :], Y_[P]


def get_mnist_dl(fpath=DOWNLOAD_DIR, to_np=False, bs=128, pm=False, shuffle=False,
                 normalize=False, flatten=False, binarize=False, y0={0, 1, 2, 3, 4}):
    (X_tr, Y_tr), (X_te, Y_te) = get_mnist(fpath, normalize=normalize, flatten=flatten, binarize=binarize, y0=y0)
    tr_dl = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=bs, shuffle=shuffle, pin_memory=pm)
    te_dl = DataLoader(TensorDataset(X_te, Y_te), batch_size=bs, pin_memory=pm)
    return tr_dl, te_dl


def get_cifar_dl(fpath=DOWNLOAD_DIR, use_cifar10=False, bs=128, shuffle=True, transform_type='none',
                 means=None, std=None, normalize=False, flatten_data=False, use_grayscale=False, nw=4, pm=False,
                 binarize=False, y0={0, 1, 2, 3, 4}):
    """data in dataloaders have has shape (B, C, W, H)"""
    d_tr, d_te = get_cifar(fpath, use_cifar10=use_cifar10, use_grayscale=use_grayscale, transform_type=transform_type,
                           normalize=normalize, means=means, std=std, flatten_data=flatten_data, binarize=binarize,
                           y0=y0)
    tr_dl = DataLoader(d_tr, batch_size=bs, shuffle=shuffle, num_workers=nw, pin_memory=pm)
    te_dl = DataLoader(d_te, batch_size=bs, num_workers=nw, pin_memory=pm)
    return tr_dl, te_dl


def get_binary_mnist(y1=0, y2=1, apply_padding=True, repeat_channels=True):
    def _make_cifar_compatible(X):
        if apply_padding: X = np.stack([np.pad(X[i][0], 2)[None, :] for i in range(len(X))])  # pad
        if repeat_channels: X = np.repeat(X, 3, axis=1)  # add channels
        return X

    binarize = lambda X, Y: get_binary_datasets(X, Y, y1=y1, y2=y2)

    tr_dl, te_dl = get_mnist_dl(normalize=False)
    Xtr, Ytr = binarize(*extract_numpy_from_loader(tr_dl))
    Xte, Yte = binarize(*extract_numpy_from_loader(te_dl))
    Xtr, Xte = map(_make_cifar_compatible, [Xtr, Xte])
    return (Xtr, Ytr), (Xte, Yte)


def get_binary_cifar(y1=3, y2=5, c={0, 1, 2, 3, 4}, use_cifar10=True):
    binarize = lambda X, Y: get_binary_datasets(X, Y, y1=y1, y2=y2)
    binary = False if y1 is not None and y2 is not None else True
    if binary: print("grouping cifar classes")
    tr_dl, te_dl = get_cifar_dl(use_cifar10=use_cifar10, shuffle=False, normalize=False, binarize=binary, y0=c)

    Xtr, Ytr = binarize(*extract_numpy_from_loader(tr_dl))
    Xte, Yte = binarize(*extract_numpy_from_loader(te_dl))
    return (Xtr, Ytr), (Xte, Yte)


def partition(X, Y, randomize=False):
    """partition randomly or using labels"""
    ni, pi = (Y == 0).nonzero()[0], (Y == 1).nonzero()[0]
    return X[pi], X[ni]


def _combine(X1, X2):
    """concatenate images from two sources"""
    X = []
    for i in range(min(len(X1), len(X2))):
        x1, x2 = X1[i], X2[i]
        x = np.concatenate((x1, x2), axis=1)
        X.append(x)
    return np.stack(X)


def np_bernoulli(p, size):
    return (np.random.rand(size) < p).astype(int)


def np_xor(a, b):
    return np.abs((a - b))  # Assumes both inputs are either 0 or 1


class DataCube(object):
    def __init__(self, Xcube):
        self.Xcube = Xcube
        self.length = Xcube.shape[0]
        self.index = 0

    def send(self, length):
        new_loc = self.index + length
        assert new_loc <= self.length, "require=%s, have=%s" % (new_loc, self.length)
        send_cube = self.Xcube[
                    self.index:new_loc]
        self.index = new_loc
        return send_cube


class OneEnv(object):
    def __init__(self, cons_ratio, env_num):
        self.cons_ratio = cons_ratio
        self.env_num = env_num
        self.assign()
        self.cpmp, self.cnmn, self.cpmn, self.cnmp = self.assign()

    def assign(self):
        ratio = self.cons_ratio
        total_num = self.env_num
        cp = total_num // 2
        cn = total_num - cp
        cpmp = int(cp * ratio)
        cnmn = int(cn * ratio)
        cpmn = cp - cpmp
        cnmp = cn - cnmn
        return cpmp, cnmn, cpmn, cnmp


class EnvsConfigure(object):
    def __init__(self, cons_ratios, train_num, test_num, train_envs_ratio, color_spurious=False):
        self.cons_ratios = cons_ratios
        self.train_envs_ratio = train_envs_ratio
        self.color_spurious = color_spurious
        self.train_num = train_num
        self.test_num = test_num
        self.env_objects = []
        self.configure()

    def configure(self):
        envs = self.cons_ratios
        train_envs_ratio = self.train_envs_ratio
        for i in range(len(envs)):
            if i != len(envs) - 1:
                if train_envs_ratio is None:
                    env_num = self.train_num // (len(envs) - 1)
                else:
                    assert sum(train_envs_ratio) == 1
                    env_num = int(self.train_num * train_envs_ratio[i])
            else:
                env_num = self.test_num
            self.env_objects.append(
                OneEnv(cons_ratio=envs[i], env_num=env_num))


def render_color(X, C):
    c1loc = (C == 1)
    c0loc = (C == 0)
    X[c1loc, 1, :, :] = 0
    X[c0loc, 0, :, :] = 0
    return X


def combine_datasets_by_envs(Xtrm, Ytrm, Xtrc, Ytrc, Xtem, Ytem, Xtec, Ytec, envs, train_num, test_num,
                             train_envs_ratio, label_noise_ratio=None, color_spurious=False):
    """combine two datasets"""
    Xm = np.concatenate([Xtrm, Xtem], axis=0)
    Ym = np.concatenate([Ytrm, Ytem], axis=0)
    Xc = np.concatenate([Xtrc, Xtec], axis=0)
    Yc = np.concatenate([Ytrc, Ytec], axis=0)
    ecf = EnvsConfigure(envs, train_num, test_num, train_envs_ratio, color_spurious=color_spurious)
    if label_noise_ratio is not None:
        if label_noise_ratio > 0:
            label_noise = np_bernoulli(
                label_noise_ratio,
                len(Yc))
            Yc = np_xor(Yc, label_noise)
    Xmp, Xmn = partition(Xm, Ym)
    Xcp, Xcn = partition(Xc, Yc)
    n = min(map(len, [Xmp, Xmn, Xcp, Xcn]))
    Xmp, Xmn, Xcp, Xcn = map(lambda Z: Z[:n], [Xmp, Xmn, Xcp, Xcn])
    train_envs = len(envs) - 1
    test_envs = 1
    train_assigns = []  # tuple: cpmp, cpmn, cnmp, cnmn
    FullX, FullY, FullG, Full_SP = None, None, None, None
    XmpCube, XmnCube, XcpCube, XcnCube = \
        DataCube(Xmp), DataCube(Xmn), DataCube(Xcp), DataCube(Xcn)
    for i in range(len(ecf.env_objects)):
        one_env = ecf.env_objects[i]
        cpmp, cnmn, cpmn, cnmp = \
            one_env.cpmp, one_env.cnmn, one_env.cpmn, one_env.cnmp

        x11 = XmpCube.send(cpmp)
        x22 = XcpCube.send(cpmp)
        Xcpmp = _combine(x11, x22)
        Xcnmn = _combine(
            XmnCube.send(cnmn),
            XcnCube.send(cnmn))
        Xcpmn = _combine(
            XmnCube.send(cpmn),
            XcpCube.send(cpmn))
        Xcnmp = _combine(
            XmpCube.send(cnmp),
            XcnCube.send(cnmp))

        Xp = np.concatenate(
            [Xcpmp, Xcpmn], axis=0)
        Sp_p = np.concatenate(
            [np.ones(len(Xcpmp)), np.zeros(len(Xcpmn))], axis=0)
        Yp = np.ones(len(Xp))
        Xn = np.concatenate(
            [Xcnmn, Xcnmp], axis=0)
        Sp_n = np.concatenate(
            [np.ones(len(Xcpmp)), np.zeros(len(Xcpmn))], axis=0)

        Yn = np.zeros(len(Xn))
        Sp = np.concatenate([Sp_p, Sp_n], axis=0)
        X = np.concatenate([Xp, Xn], axis=0)
        Y = np.concatenate([Yp, Yn], axis=0)
        if color_spurious:
            color_noise = np_bernoulli(
                1 - one_env.cons_ratio,
                len(Y))
            C = np_xor(Y, color_noise)
            X = render_color(X, C)
        G = np.ones_like(Y) * i
        if FullX is None:
            FullX, FullY, FullG, Full_SP = X, Y, G, Sp
        else:
            FullX = np.concatenate(
                [FullX, X], axis=0)
            FullY = np.concatenate(
                [FullY, Y], axis=0)
            FullG = np.concatenate(
                [FullG, G], axis=0)
            Full_SP = np.concatenate(
                [Full_SP, Sp], axis=0)
    P = np.random.permutation(len(FullX))
    FullX, FullY, FullG = FullX[P], FullY[P], FullG[P]
    Full_SP = Full_SP[P]
    return FullX, FullY, FullG, Full_SP


def get_mnist_cifar_env(mnist_classes=(0, 1), cifar_classes=(1, 9), c={0, 1, 2, 3, 4}, randomize_mnist=False,
                        randomize_cifar=False, train_num=None, test_num=None, cons_ratios=None, train_envs_ratio=None,
                        label_noise_ratio=None, color_spurious=False, oracle=0):
    np.random.seed(1)
    random.seed(1)  # Fix the random seed of dataset
    y1, y2 = mnist_classes
    (Xtrm, Ytrm), (Xtem, Ytem) = get_binary_mnist(y1=y1, y2=y2)
    if oracle:
        Xtrm = np.ones_like(Xtrm)
    y1, y2 = (None, None) if cifar_classes is None else cifar_classes
    (Xtrc, Ytrc), (Xtec, Ytec) = get_binary_cifar(c=c, y1=y1, y2=y2)

    FullX, FullY, FullG, FullSP = combine_datasets_by_envs(Xtrm, Ytrm, Xtrc, Ytrc, Xtem, Ytem, Xtec, Ytec,
                                                           envs=cons_ratios, train_num=train_num, test_num=test_num,
                                                           train_envs_ratio=train_envs_ratio,
                                                           label_noise_ratio=label_noise_ratio,
                                                           color_spurious=color_spurious)
    return FullX, FullY, FullG, FullSP


def get_transform_cub(transform_data_to_standard, train, augment_data):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform


class SubDataset(object):
    def __init__(self, x_array, y_array, env_array, transform, sp_array=None):
        self.x_array = x_array
        self.y_array = y_array[:, None]
        self.env_array = env_array[:, None]
        self.sp_array = sp_array[:, None]
        self.transform = transform
        assert len(self.x_array) == len(self.y_array)
        assert len(self.y_array) == len(self.env_array)

    def __len__(self):
        return len(self.x_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.env_array[idx]
        if self.sp_array is not None:
            sp = self.sp_array[idx]
        else:
            sp = None
        img = self.x_array[idx]
        img = (img * 255).astype(np.uint8)
        img = img.transpose(1, 2, 0)
        img = Image.fromarray(img)
        x = self.transform(img)

        return x, y, g, sp


class SpuriousValDataset(Dataset):
    def __init__(self, val_dataset):
        self.val_dataset = val_dataset

    def __len__(self):
        return len(self.val_dataset)

    def __getitem__(self, idx):
        x, y, g, sp = self.val_dataset.__getitem__(idx)
        g = g * 0
        return x, y, g, sp


class CifarMnistSpuriousDataset(Dataset):
    def __init__(self, train_num, test_num, cons_ratios, cifar_classes=(1, 9), train_envs_ratio=None,
                 label_noise_ratio=None, augment_data=True, color_spurious=False, transform_data_to_standard=1,
                 oracle=0):
        self.cons_ratios = cons_ratios
        self.train_num = train_num
        self.test_num = test_num
        self.train_envs_ratio = train_envs_ratio
        self.augment_data = augment_data
        self.oracle = oracle
        self.x_array, self.y_array, self.env_array, self.sp_array = \
            get_mnist_cifar_env(
                train_num=train_num,
                test_num=test_num,
                cons_ratios=cons_ratios,
                train_envs_ratio=train_envs_ratio,
                label_noise_ratio=label_noise_ratio,
                cifar_classes=cifar_classes,
                color_spurious=color_spurious,
                oracle=oracle)
        self.feature_dim = self.x_array.reshape([self.x_array.shape[0], -1]).shape[1]
        self.transform_data_to_standard = transform_data_to_standard
        self.y_array = self.y_array.astype(np.int64)
        self.split_array = self.env_array
        self.n_train_envs = len(self.cons_ratios) - 1
        self.split_dict = {
            "train": range(len(self.cons_ratios) - 1),
            "val": [len(self.cons_ratios) - 1],
            "test": [len(self.cons_ratios) - 1]}
        self.n_classes = 2
        self.train_transform = get_transform_cub(transform_data_to_standard=self.transform_data_to_standard, train=True,
                                                 augment_data=self.augment_data)
        self.eval_transform = get_transform_cub(transform_data_to_standard=self.transform_data_to_standard, train=False,
                                                augment_data=False)

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.env_array[idx]

        img = self.x_array[idx]
        sp = self.sp_array[idx]
        # Figure out split and transform accordingly
        if self.split_array[idx] in self.split_dict['train']:
            img = self.train_transform(img)
        elif self.split_array[idx] in self.split_dict['val'] + self.split_dict['test']:
            img = self.eval_transform(img)
        x = img

        return x, y, g, sp

    def get_splits(self, splits, train_frac=1.0):
        subsets = []
        for split in splits:
            assert split in ('train', 'val', 'test'), split + ' is not a valid split'
            mask = np.isin(self.split_array, self.split_dict[split])
            num_split = np.sum(mask)
            indices = np.where(mask)[0]
            if split == "train":
                subsets.append(
                    SubDataset(
                        x_array=self.x_array[indices],
                        y_array=self.y_array[indices],
                        env_array=self.env_array[indices],
                        sp_array=self.sp_array[indices],
                        transform=self.train_transform
                    ))
            else:
                subsets.append(
                    SpuriousValDataset(
                        SubDataset(
                            x_array=self.x_array[indices],
                            y_array=self.y_array[indices],
                            env_array=self.env_array[indices],
                            sp_array=self.sp_array[indices],
                            transform=self.train_transform
                        )))

        self.subsets = subsets
        return tuple(subsets)


def get_data_loader_cifarminst(batch_size, train_num, test_num, cons_ratios, train_envs_ratio, label_noise_ratio=None,
                               augment_data=True, cifar_classes=(1, 9), color_spurious=False,
                               transform_data_to_standard=1, oracle=0):
    np.random.seed(1)
    random.seed(1)
    spdc = CifarMnistSpuriousDataset(
        train_num=train_num,
        test_num=test_num,
        cons_ratios=cons_ratios,
        train_envs_ratio=train_envs_ratio,
        label_noise_ratio=label_noise_ratio,
        augment_data=augment_data,
        cifar_classes=cifar_classes,
        color_spurious=color_spurious,
        transform_data_to_standard=transform_data_to_standard,
        oracle=oracle)
    train_dataset, val_dataset, test_dataset = spdc.get_splits(
        splits=['train', 'val', 'test'])
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)
    return spdc, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


class LYDataProvider(object):
    def __init__(self):
        pass

    def preprocess_data(self):
        pass

    def fetch_train(self):
        pass

    def fetch_test(self):
        pass


class CIFAR_LYPD(LYDataProvider):
    def __init__(self, batch_size, cifar_classess=(0, 5)):
        super(CIFAR_LYPD, self).__init__()
        self.cifar_classes = cifar_classess
        self.batch_size = batch_size
        np.random.seed(1)
        random.seed(1)  # Fix the random seed of dataset
        self.preprocess_data()

    def preprocess_data(self):
        train_num = 10000
        test_num = 1000
        cons_list = [0.999, 0.7, 0.1]
        train_envs = len(cons_list) - 1
        ratio_list = [1. / train_envs] * (train_envs)
        spd, self.train_loader, self.val_loader, self.test_loader, self.train_data, self.val_data, self.test_data = \
            get_data_loader_cifarminst(
                batch_size=self.batch_size,
                train_num=train_num,
                test_num=test_num,
                cons_ratios=cons_list,
                train_envs_ratio=ratio_list,
                label_noise_ratio=0.25,
                color_spurious=False,
                transform_data_to_standard=0,
                oracle=0,
                cifar_classes=self.cifar_classes
            )
        self.train_loader_iter = iter(self.train_loader)

    def fetch_train(self, device="cpu"):
        try:
            batch_data = self.train_loader_iter.__next__()
        except:
            self.train_loader_iter = iter(self.train_loader)
            batch_data = self.train_loader_iter.__next__()
        batch_data = tuple(t.to(device) for t in batch_data)
        x, y, g, sp = batch_data
        return x, y.float().to(device), g, sp

    def fetch_test(self, device="cpu"):
        ds = self.test_data.val_dataset
        batch = ds.x_array, ds.y_array, ds.env_array, ds.sp_array
        batch = tuple(
            torch.Tensor(t).to(device)
            for t in batch)
        x, y, g, sp = batch
        return x, y.float(), g, sp

    def test_batchs(self):
        return math.ceil(self.test_data.val_dataset.x_array.shape[0] / self.batch_size)


if __name__ == "__main__":
    dp = CIFAR_LYPD(256)
    test_batch_num = 1
    train_x, train_y, train_g, train_c = dp.fetch_train()
    print(train_x.shape)
    print(train_y.shape)
    print(train_g.shape)
    print(train_c.shape)

    torch.save(train_x, "../data/example/train_x.pth.tar")
    torch.save(train_y, "../data/example/train_y.pth.tar")
    torch.save(train_g, "../data/example/train_g.pth.tar")
    torch.save(train_c, "../data/example/train_c.pth.tar")
