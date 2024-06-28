import torch
import torchvision.transforms as transforms
from torch import nn, optim
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import copy
from models.resnets import resnet20s



def cosine_similarity_torch(A, B):
    cosine_sim = torch.dot(A, B) / (torch.norm(A) * torch.norm(B))
    return cosine_sim


def train(model, device, train_loader, optimizer, criterion):
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

def eval(model, test_loader, criterion=nn.CrossEntropyLoss()):
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



# Test function
if __name__ == "__main__":

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

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

    # Experiment 1, Task 1: [255,0,0], flip: False, Task 2: [251,44,0], flip: False
    # Experiment 2, Task 1: [255,0,0], flip: False, Task 2: [0,255,0], flip: False
    # Experiment 3, Task 1: [255,0,0], flip: False, Task 2: [251,44,0], flip: True

    for color1, color2, flip1, flip2 in [([255, 0, 0], [251,44,0], False, False), ([255, 0, 0], [0,255,0], False, False),
                                    ([255, 0, 0], [251, 44, 0], False, True)]:

        # Usage example:
        train_set_task1 = CustomMNIST(root='./data', train=True, download=True, color=color1, flip_label=flip1)
        train_set_task2 = CustomMNIST(root='./data', train=True, download=True, color=color2, flip_label=flip2)

        test_set_task1 = CustomMNIST(root='./data', train=False, download=True, color=color1, flip_label=flip1)
        test_set_task2 = CustomMNIST(root='./data', train=False, download=True, color=color2, flip_label=flip2)

        # Data loaders
        train_loader_task1 = DataLoader(train_set_task1, batch_size=256, shuffle=True)
        train_loader_task2 = DataLoader(train_set_task2, batch_size=256, shuffle=True)
        test_loader_task1 = DataLoader(test_set_task1, batch_size=256, shuffle=False)
        test_loader_task2 = DataLoader(test_set_task2, batch_size=256, shuffle=False)

        # Model
        model_init = resnet20s(num_classes=2).to(device)
        model_task1 = copy.deepcopy(model_init).to(device)
        model_task2 = copy.deepcopy(model_init).to(device)

        for param in model_task1.fc.parameters():
            param.requires_grad = False
        for param in model_task2.fc.parameters():
            param.requires_grad = False

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer_task1 = optim.Adam(model_task1.parameters(), lr=0.001)
        optimizer_task2 = optim.Adam(model_task2.parameters(), lr=0.001)

        # Train red
        for epoch in tqdm(range(1)):
            train(model_task1, device, train_loader_task1, optimizer_task1, criterion)
            train(model_task2, device, train_loader_task2, optimizer_task2, criterion)

        test_acc_task1, test_loss_task1 = eval(model_task1, test_loader_task1, nn.CrossEntropyLoss())
        test_acc_task2, test_loss_task2 = eval(model_task2, test_loader_task2, nn.CrossEntropyLoss())

        print(f'Task 1 accuracy: {test_acc_task1}, Task 2 accuracy: {test_acc_task2}')

        model_init_params = torch.cat([p.view(-1) for p in model_init.parameters()])
        model_task1_params = torch.cat([p.view(-1) for p in model_task1.parameters()])
        model_task_vector1 = model_task1_params - model_init_params
        model_task2_params = torch.cat([p.view(-1) for p in model_task2.parameters()])
        model_task_vector2 = model_task2_params - model_init_params

        # Test
        results = []
        for lmbda in range(-10, 11, 2):
            lmbda /= 10.0
            # lmbda = 0
            new_model = copy.deepcopy(model_task1).to(device)
            # Update the model parameters
            start = 0
            for param in new_model.parameters():
                end = start + param.numel()
                param.data.add_(lmbda * model_task_vector2[start: end].view(param.size()))
                start = end

            # Evaluate the new model
            test_acc_task1, test_loss_task1 = eval(new_model, test_loader_task1, nn.CrossEntropyLoss())
            test_acc_task2, test_loss_task2 = eval(new_model, test_loader_task2, nn.CrossEntropyLoss())
            print(f'Lambda: {lmbda}, Task 1 accuracy: {test_acc_task1}, Task 2 accuracy: {test_acc_task2}')

            results.append((lmbda, test_acc_task1, test_acc_task2))
