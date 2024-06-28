import torch
import torchvision.transforms as transforms
from torch import nn, optim
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import numpy as np
import random
from models.resnets import resnet20s
import copy


class SimpleMLP(nn.Module):
    def __init__(self, num_classes, input_size=32):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            # First hidden layer
            nn.Linear(3 * input_size * input_size, 128 * 4),
            nn.BatchNorm1d(128 * 4),
            nn.ReLU(),

            # Second hidden layer
            nn.Linear(128 * 4, 128 * 2),
            nn.BatchNorm1d(128 * 2),
            nn.ReLU(),

            # Third hidden layer
            nn.Linear(128 * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # Output layer
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.layers(x)


def cosine_similarity_manual(A, B):
    # Calculate dot product
    dot_product = sum(a * b for a, b in zip(A, B))

    # Calculate magnitude of A
    magnitude_A = sum(a ** 2 for a in A) ** 0.5

    # Calculate magnitude of B
    magnitude_B = sum(b ** 2 for b in B) ** 0.5

    # Calculate cosine similarity
    cosine_sim = dot_product / (magnitude_A * magnitude_B)

    return cosine_sim


def train(epoch, model, device, train_loader, optimizer, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Convert labels to 0 and 1 for binary classification (-1 becomes 0, 1 stays 1)
        target = (target + 1) // 2
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % 100 == 0:
        #     print(
        #         f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


# Test function
def eval(model, test_loader, criterion, color_id=0):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # Convert labels to 0 and 1 for testing
            target = (target + 1) // 2
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    return test_acc, test_loss


if __name__ == "__main__":

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Custom MNIST Dataset
    class CustomMNIST(MNIST):
        def __init__(self, *args, color=(255, 0, 0), flip_label=False, **kwargs):
            # Ensure no transform is applied by the parent class, handle it in __getitem__
            kwargs['transform'] = None
            super().__init__(*args, **kwargs)
            # Convert color to a 0-1 range
            self.color = np.array(color, dtype=np.float32) / 255.0
            self.flip_label = flip_label

        def __getitem__(self, index):
            # Retrieve an image and its label from the standard MNIST dataset without any transform applied
            img, target = super(CustomMNIST, self).__getitem__(index)
            # Modify the label: 1 if odd, -1 if even
            if self.flip_label:
                target = -1 if target % 2 != 0 else 1
            else:
                target = 1 if target % 2 != 0 else -1

            # Convert the grayscale image to RGB
            img = img.convert("RGB")

            # Convert the PIL image to a NumPy array
            img_array = np.array(img)

            # Apply the new color to non-black pixels
            mask = img_array[:, :, 0] > 0  # Get the mask of the digit
            img_array[mask] = (self.color * 255).astype(np.uint8)  # Apply color to the digit

            # Convert the NumPy array back to a PIL image
            img = transforms.ToPILImage()(img_array)

            # Now apply the transformation for model input
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ])

            img = transform(img)
            return img, target

    results = []

    red_color = (255, 0, 0)

    # Load datasets with custom labels and colors
    train_set = CustomMNIST(root='./data', train=True, download=True, color=red_color)
    test_set = CustomMNIST(root='./data', train=False, download=True, color=red_color)

    # Data loaders
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    # Model
    model_init = resnet20s(num_classes=2).to(device)
    red_model = copy.deepcopy(model_init)
    red_model.to(device)
    for params in red_model.fc.parameters():
        params.requires_grad = False

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(red_model.parameters(), lr=0.001)

    # Train red
    for epoch in range(0, 2):  # 2 epochs are enough
        train(epoch, red_model, device, train_loader, optimizer, criterion)
    test_acc, test_loss = eval(red_model, test_loader, criterion, color_id=0)
    results.append({"test_acc": test_acc, "test_loss": test_loss, "color": red_color, "color_cosine_similarity": 1.0, "theta_cosine_similarity": 1.0})
    print(
        f'Color ID: {0}, \nTest set: Average loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%, Color: {red_color}, Color cosine similarity: 1.0, Theta cosine similarity: 1.0')

    for sample_id in range(100):
        # Random color generation
        random_color = tuple(random.randint(0, 255) for _ in range(3))
        cosine_similarity_value = np.sum(np.array(red_color) * np.array(random_color)) / (np.linalg.norm(red_color) * np.linalg.norm(random_color))

        # Load datasets with custom labels and colors
        # Load datasets with custom labels and colors
        train_set = CustomMNIST(root='./data', train=True, download=True, color=random_color)
        test_set = CustomMNIST(root='./data', train=False, download=True, color=random_color)

        # Data loaders
        train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

        # Model
        model = copy.deepcopy(model_init)
        model.to(device)
        for params in model.fc.parameters():
            params.requires_grad = False

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(0, 2):  # 2 epochs are enough
            train(epoch, model, device, train_loader, optimizer, criterion)
        test_acc, test_loss = eval(model, test_loader, criterion, color_id=sample_id + 1)
        # Calculate the cosine similarity between the parameters of red_model and the current model
        # First collect the parameters of each model and flatten them
        model_init_params = torch.cat([p.view(-1) for p in model_init.parameters()])
        red_model_params = torch.cat([p.view(-1) for p in red_model.parameters()])
        red_model_task_vector = red_model_params - model_init_params
        model_params = torch.cat([p.view(-1) for p in model.parameters()])
        model_task_vector = model_params - model_init_params
        # Calculate the cosine similarity
        theta_cosine_similarity = cosine_similarity_manual(red_model_task_vector, model_task_vector)
        results.append({"test_acc": test_acc, "test_loss": test_loss, "color": random_color, "color_cosine_similarity": cosine_similarity_value, "theta_cosine_similarity": theta_cosine_similarity})
        print(
            f'Color ID: {sample_id + 1}, \nTest set: Average loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%, Color: {random_color}, Color cosine similarity: {cosine_similarity_value}, Theta cosine similarity: {theta_cosine_similarity}')
        # Save the results
        torch.save(results, "results/exp2.pth")



