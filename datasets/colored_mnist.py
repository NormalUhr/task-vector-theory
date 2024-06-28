import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets

from datasets.data_utils import make_environment


class CMNIST:
    def __init__(self, args):
        self.args = args

    def data_loaders(self, **kwargs):

        # if you don't provide class imbalance rate, then balance the class.
        if len(self.args.training_class_env) == 0:
            self.args.training_class_env = [0.5 for _ in range(len(self.args.training_env))]
        # if you only provide one imbalance rate, then use this for all training settings.
        elif len(self.args.training_class_env) == 1:
            p = self.args.training_class_env[0]
            self.args.training_class_env = [p for i in range(len(self.args.training_env))]
        else:
            assert len(self.args.training_class_env) == len(self.args.training_env)

        if len(self.args.training_color_env) == 0:
            use_color = False
            self.args.training_color_env = [0.5 for _ in range(len(self.args.training_env))]
            # if you only provide one imbalance rate, then use this for all training settings.
        elif len(self.args.training_color_env) == 1:
            use_color = True
            p = self.args.training_color_env[0]
            self.args.training_color_env = [p for _ in range(len(self.args.training_env))]
        else:
            use_color = True
            assert len(self.args.training_color_env) == len(self.args.training_env)

        mnist = datasets.MNIST(self.args.data_dir, train=True, download=True)
        mnist_test = datasets.MNIST('~/datasets/mnist', train=False, download=True)

        mnist_train = (mnist.data[:50000], mnist.targets[:50000])
        mnist_val = (mnist.data[50000:], mnist.targets[50000:])
        mnist_test = (mnist_test.data, mnist_test.targets)

        rng_state = np.random.get_state()
        np.random.shuffle(mnist_train[0].numpy())
        np.random.set_state(rng_state)
        np.random.shuffle(mnist_train[1].numpy())
        env_sets = []

        for i, env_p in enumerate(self.args.training_env):
            env_sets.append(make_environment(mnist_train[0][i::len(self.args.training_env)],
                                             mnist_train[1][i::len(self.args.training_env)], env_p,
                                             label_flip_p=self.args.label_flip_p,
                                             class_p=self.args.training_class_env[i],
                                             color_p=self.args.training_color_env[i],
                                             use_color=use_color))

        train_loader = []

        for env_set in env_sets:
            train_ld = DataLoader(
                env_set,
                batch_size=self.args.batch_size,
                shuffle=True,
                **kwargs
            )
            train_loader.append(train_ld)

        val_set = make_environment(mnist_val[0], mnist_val[1], self.args.test_env,
                                   label_flip_p=self.args.label_flip_p,
                                   class_p=0.5, color_p=0.5, use_color=use_color)

        val_loader = DataLoader(
            val_set,
            batch_size=self.args.batch_size,
            shuffle=True,
            **kwargs
        )

        test_env_loaders = []
        for test_p in range(5, 100, 5):
            test_p = test_p / 100.
            test_env_loaders.append(
                make_environment(mnist_test[0], mnist_test[1], test_p, label_flip_p=self.args.label_flip_p,
                                 class_p=0.5, color_p=0.5, use_color=use_color))

        test_loader = []

        for env_set in test_env_loaders:
            test_ld = DataLoader(
                env_set,
                batch_size=self.args.batch_size,
                shuffle=True,
                **kwargs
            )
            test_loader.append(test_ld)

        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    from args import parse_args

    args = parse_args()
    dataset = CMNIST(args)

    train_loader, val_loader, test_loader = dataset.data_loaders()

    total = 0
    for loader in train_loader:
        for images, labels in loader:
            total += images.shape[0]

    print(total)

    total = 0
    for loader in test_loader:
        for images, labels in loader:
            total += images.shape[0]

    print(total)
