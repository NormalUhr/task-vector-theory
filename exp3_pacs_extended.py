import torch
import torchvision.transforms as transforms
from torch import nn, optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import copy
import random
import numpy as np

from torch.utils.data import Dataset
import os
from PIL import Image

from sklearn.model_selection import train_test_split

class PACSDataset(Dataset):
    def __init__(self, root_dir, domain_index, train=True, split_ratio=0.9, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.domain_index = domain_index
        self.train = train
        self.split_ratio = split_ratio

        self.domains = ['art_painting', 'cartoon', 'photo', 'sketch']
        self.classes = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Set the domain directory
        self.domain_dir = os.path.join(self.root_dir, self.domains[self.domain_index])

        # Load all images and labels
        images = []
        labels = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.domain_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith('.jpg') or img_name.lower().endswith('.png'):
                    img_path = os.path.join(cls_dir, img_name)
                    images.append(img_path)
                    labels.append(self.class_to_idx[cls_name])

        # Split the data
        self.images, self.labels = self._split_data(images, labels)

    def _split_data(self, images, labels):
        # Split data into train and test sets
        train_images, test_images, train_labels, test_labels = train_test_split(
            images, labels, train_size=self.split_ratio, random_state=42, shuffle=True, stratify=labels)

        if self.train:
            return train_images, train_labels
        else:
            return test_images, test_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def cosine_similarity_torch(A, B):
    cosine_sim = torch.dot(A, B) / (torch.norm(A) * torch.norm(B))
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


# Test function
def eval(model, test_loader, criterion):
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
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({test_acc:.0f}%)')
    return test_acc, test_loss


def setup_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":

    setup_seed(42)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model_init = resnet18(pretrained=True).to(device)
    model_domain1 = copy.deepcopy(model_init).to(device)
    model_domain2 = copy.deepcopy(model_init).to(device)
    model_domain3 = copy.deepcopy(model_init).to(device)
    model_domain4 = copy.deepcopy(model_init).to(device)
    models = [model_domain1, model_domain2, model_domain3, model_domain4]
    for model in models:
        for param in model.fc.parameters():
            param.requires_grad = False

    # Usage Example:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    for domain_index in range(4):
        # Load datasets with custom labels and colors
        train_set = PACSDataset(root_dir='../data/PACS/', domain_index=domain_index, transform=transform, train=True)
        test_set = PACSDataset(root_dir='../data/PACS/', domain_index=domain_index, transform=transform, train=False)

        # Data loaders
        train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(models[domain_index].parameters(), lr=0.001)
        if os.path.exists("results/exp3_extended.pth"):
            epochs = 0
        else:
            epochs = 20
        # Cosine learning rate
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

        # Train red
        for epoch in range(0, epochs):
            train(epoch, models[domain_index], device, train_loader, optimizer, criterion)
            test_acc, test_loss = eval(models[domain_index], test_loader, criterion)
            scheduler.step()

    if os.path.exists("results/exp3_extended.pth"):
        model_task_vectors = torch.load("results/exp3_extended.pth")["model_ckpts"]
        model_init_params = model_task_vectors[-1]
    else:
        model_task_vectors = []
        # First collect the parameters of each model and flatten them
        model_init_params = torch.cat([p.view(-1) for p in model_init.parameters()])

        for model in models:
            model_params = torch.cat([p.view(-1) for p in model.parameters()])
            model_task_vector = model_params - model_init_params
            model_task_vectors.append(model_task_vector)

        model_task_vectors.append(model_init_params)

        torch.save({"model_ckpts": model_task_vectors}, "results/exp3_extended.pth")

    # Calculate the cosine similarity between the task vectors
    cosine_sim_matrix = torch.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            cosine_sim_matrix[i, j] = cosine_similarity_torch(model_task_vectors[i], model_task_vectors[j]).detach().cpu()
            print(f"cosine similarity between model {i} and model {j}: {cosine_sim_matrix[i, j]}")

    # Experiment 1
    task_vector_combination = model_task_vectors[1] + model_task_vectors[2] + model_task_vectors[3]
    for lmbda in range(2, 102, 2):
        lmbda = lmbda / 100
        model_comb = copy.deepcopy(model_init)
        model_comb_params = model_init_params + lmbda * task_vector_combination
        # Update the model parameters
        start = 0
        for param in model_comb.parameters():
            end = start + param.numel()
            param.data = model_comb_params[start:end].view(param.size())
            start = end

        test_set = PACSDataset(root_dir='../data/PACS/', domain_index=0, transform=transform, train=False)
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

        test_acc, test_loss = eval(model_comb, test_loader, nn.CrossEntropyLoss())
        print(f"Experiment 1: Lambda: {lmbda}, Test Accuracy: {test_acc}")

    # Experiment 2
    task_vector_combination = 0.86 * model_task_vectors[1] + 1.124 * model_task_vectors[2] + 0.374 * model_task_vectors[3]
    for lmbda in range(2, 21, 2):
        model_comb = copy.deepcopy(model_init)
        model_comb_params = model_init_params + lmbda * task_vector_combination / 10
        # Update the model parameters
        start = 0
        for param in model_comb.parameters():
            end = start + param.numel()
            param.data = model_comb_params[start:end].view(param.size())
            start = end

        test_set = PACSDataset(root_dir='../data/PACS/', domain_index=0, transform=transform, train=False)
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

        test_acc, test_loss = eval(model_comb, test_loader, nn.CrossEntropyLoss())
        print(f"Experiment 2: Lambda: {lmbda}, Test Accuracy: {test_acc}")