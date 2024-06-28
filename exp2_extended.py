import timm
import torch
import torchvision.transforms as transforms
from torch import nn, optim
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import copy


def cosine_similarity_torch(A, B):
    cosine_sim = torch.dot(A, B) / (torch.norm(A) * torch.norm(B))
    return cosine_sim


def train(model, device, train_loader, optimizer, criterion):
    model.train()
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        # Convert labels to 0 and 1 for binary classification (-1 becomes 0, 1 stays 1)
        target = (target + 1) // 2
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


# Test function
if __name__ == "__main__":

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Custom MNIST Dataset
    class CustomMNIST(MNIST):
        def __init__(self, *args, color_1=(255, 0, 0), color_2=(0, 255, 0), beta=0.5, **kwargs):
            super().__init__(*args, **kwargs)
            self.color_1 = np.array(color_1, dtype=np.float32) / 255.0  # Normalize the color vector
            self.color_2 = np.array(color_2, dtype=np.float32) / 255.0
            self.beta = beta
            self.distribution = self._generate_distribution()

        def _generate_distribution(self):
            # Calculate color distribution for each class
            total_items = len(self.targets)
            targets_np = np.array(self.targets)
            odd_indices = np.where(targets_np % 2 != 0)[0]
            even_indices = np.where(targets_np % 2 == 0)[0]

            num_odd_color_1 = int(len(odd_indices) * self.beta / (1 + self.beta))
            num_even_color_2 = int(len(even_indices) * self.beta / (1 + self.beta))

            odd_color_1_indices = np.random.choice(odd_indices, num_odd_color_1, replace=False)
            even_color_2_indices = np.random.choice(even_indices, num_even_color_2, replace=False)

            color_distribution = np.zeros(total_items, dtype=int)
            color_distribution[odd_color_1_indices] = 1  # Color 1 for odd
            color_distribution[even_color_2_indices] = 2  # Color 2 for even

            return color_distribution

        def __getitem__(self, index):
            img, target = super(CustomMNIST, self).__getitem__(index)
            target = 1 if target % 2 != 0 else -1  # Modify label

            # Choose color based on distribution
            color = self.color_1 if self.distribution[index] == 1 else self.color_2
            if target == -1:
                color = self.color_2 if self.distribution[index] == 2 else self.color_1

            # Convert grayscale image to RGB and apply color
            img = img.convert("RGB")
            img_array = np.array(img)
            mask = img_array[:, :, 0] > 0
            img_array[mask] = (color * 255).astype(np.uint8)
            img = transforms.ToPILImage()(img_array)

            # Define transformations including resize and normalization
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            img = transform(img)
            return img, target

    # Experiment 1, Task 1: beta = 0.8, Task 2: beta = 1
    # Experiment 2, Task 1: beta = 0.8, Task 2: beta = 0.5
    # Experiment 3, Task 1: beta = 0.8, Task 2: beta = 0
    # Experiment 4, Task 1: beta = 1, Task 2: beta = 0

    for beta1, beta2 in [(0.8, 1.0), (0.8, 0.5), (0.8, 0.0), (1.0, 0.0)]:

        # Usage example:
        train_set_task1 = CustomMNIST(root='./data', train=True, download=True,
                                color_1=(255, 0, 0), color_2=(0, 255, 0),
                                beta=beta1)
        train_set_task2 = CustomMNIST(root='./data', train=True, download=True,
                                color_1=(255, 0, 0), color_2=(0, 255, 0),
                                beta=beta2)

        # Data loaders
        train_loader_task1 = DataLoader(train_set_task1, batch_size=256, shuffle=True)
        train_loader_task2 = DataLoader(train_set_task2, batch_size=256, shuffle=True)

        # Model
        model_init = timm.create_model("vit_tiny_patch16_224", pretrained=True).to(device)
        model_task1 = copy.deepcopy(model_init).to(device)
        model_task2 = copy.deepcopy(model_init).to(device)

        for param in model_task1.head.parameters():
            param.requires_grad = False
        for param in model_task2.head.parameters():
            param.requires_grad = False

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer_task1 = optim.Adam(model_task1.parameters(), lr=0.001)
        optimizer_task2 = optim.Adam(model_task2.parameters(), lr=0.001)

        # Train red
        for epoch in tqdm(range(0, 5)):  # 2 epochs are enough
            train(model_task1, device, train_loader_task1, optimizer_task1, criterion)
            train(model_task2, device, train_loader_task2, optimizer_task2, criterion)

        model_init_params = torch.cat([p.view(-1) for p in model_init.parameters()])
        model_task1_params = torch.cat([p.view(-1) for p in model_task1.parameters()])
        model_task_vector1 = model_task1_params - model_init_params
        model_task2_params = torch.cat([p.view(-1) for p in model_task2.parameters()])
        model_task_vector2 = model_task2_params - model_init_params

        # Calculate the cosine similarity
        theta_cosine_similarity = cosine_similarity_torch(model_task_vector1, model_task_vector2)
        print(f'Theta cosine similarity: {theta_cosine_similarity}')



