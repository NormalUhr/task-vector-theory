import torch
from torch.utils.data import Dataset
from torchvision import transforms


class EnvDataset(Dataset):
    def __init__(self, images, labels):

        assert len(images) == len(labels)

        self.tr_train = [
            transforms.RandomCrop(28, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        self.tr_test = [transforms.ToTensor()]

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # image = self.tr_train(self.images[index])
        image = self.images[index]
        label = self.labels[index]

        return image, label


# Build environments
def make_environment(images, labels, e, label_flip_p, class_p, color_p, use_color):
    def torch_bernoulli(p, size):
        return (torch.rand(size) < p).float()

    def torch_xor(a, b):
        return (a - b).abs()  # Assumes both inputs are either 0 or 1

    # 2x subsample for computational convenience
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    # Assign a binary label based on the digit; flip label with probability 0.25
    labels = (labels < 5).float()

    label0_indices = (labels == 0).nonzero().squeeze()
    label1_indices = (labels == 1).nonzero().squeeze()

    class_0 = class_p * 2 if class_p <= 0.5 else 1.0
    class_1 = (1 - class_p) * 2 if (1 - class_p) <= 0.5 else 1.0

    indices = torch.cat([label0_indices[:int(class_0 * len(label0_indices))],
                         label1_indices[:int(class_1 * len(label1_indices))]]).sort().values

    images = images[indices]
    labels = labels[indices]

    # flip label with probability 0.25
    labels = torch_xor(labels, torch_bernoulli(label_flip_p, len(labels)))
    # Assign a color based on the label; flip the color with probability color_flip
    colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
    # Apply the color to the image by zeroing out the other color channel

    if use_color:
        # This requires MLP with 4 * 14 * 14 input.
        colors_0s = (colors == 0.0).nonzero()
        colors_1s = (colors == 1.0).nonzero()

        colors_branch0 = torch_bernoulli(color_p, len(colors_0s))
        colors_branch1 = torch_bernoulli(1 - color_p, len(colors_1s))

        all_colors0 = torch.scatter(torch.zeros_like(colors), 0, colors_0s.squeeze(1), colors_branch0)
        all_colors1 = torch.scatter(torch.zeros_like(colors), 0, colors_0s.squeeze(1), 1 - colors_branch0)

        all_colors2 = torch.scatter(torch.zeros_like(colors), 0, colors_1s.squeeze(1), colors_branch1)
        all_colors3 = torch.scatter(torch.zeros_like(colors), 0, colors_1s.squeeze(1), 1 - colors_branch1)

        id_0 = torch.stack([images, images], dim=1)
        id_1 = torch.stack([images, images], dim=1)
        id_2 = torch.stack([images, images], dim=1)
        id_3 = torch.stack([images, images], dim=1)

        id_0[torch.tensor(range(len(id_0))), all_colors0.long(), :, :] *= 0
        id_1[torch.tensor(range(len(id_1))), all_colors1.long(), :, :] *= 0
        id_2[torch.tensor(range(len(id_2))), all_colors2.long(), :, :] *= 0
        id_3[torch.tensor(range(len(id_3))), all_colors3.long(), :, :] *= 0

        images = torch.stack([id_0[:, 0], id_1[:, 0], id_2[:, 0], id_3[:, 0]], dim=1)

        del id_0, id_1, id_2, id_3

    else:
        images = torch.stack([images, images], dim=1)
        images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0

    labels = labels[:, None].type(torch.LongTensor)
    images = images.float() / 255.

    # return {
    #     'images': (images.float() / 255.).to(device),
    #     'labels': labels[:, None].to(device)
    # }

    return EnvDataset(images, labels)
