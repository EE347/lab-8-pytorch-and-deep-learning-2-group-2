import time
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataset import TeamMateDataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

def train_and_evaluate(model, criterion, trainloader, testloader, device, loss_name):
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    best_train_loss = 1e9
    train_losses = []
    test_losses = []

    for epoch in range(1, 6):  # Example with 3 epochs; adjust as needed
        t = time.time_ns()
        model.train()
        train_loss = 0

        for i, (images, labels) in tqdm(enumerate(trainloader), total=len(trainloader), leave=False):
            images = train_transform(images)
            images = images.reshape(-1, 3, 64, 64).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # For NLLLoss, apply log_softmax to outputs
            if loss_name == "NLLLoss":
                outputs = F.log_softmax(outputs, dim=1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(testloader, total=len(testloader), leave=False):
            images = images.reshape(-1, 3, 64, 64).to(device)
            labels = labels.to(device)
            outputs = model(images)

            # For NLLLoss, apply log_softmax to outputs
            if loss_name == "NLLLoss":
                outputs = F.log_softmax(outputs, dim=1)

            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Epoch: {epoch}, Train Loss: {train_loss / len(trainloader):.4f}, Test Loss: {test_loss / len(testloader):.4f}, Test Accuracy: {correct / total:.4f}, Time: {(time.time_ns() - t) / 1e9:.2f}s')

        train_losses.append(train_loss / len(trainloader))
        test_losses.append(test_loss / len(testloader))

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), f'lab8/best_model_{loss_name}.pth')

        torch.save(model.state_dict(), f'lab8/current_model_{loss_name}.pth')

    # Plot and save loss curves for each criterion
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{loss_name} Loss Curve')
    plt.legend()
    plt.savefig(f'lab8/task5_{loss_name}_loss_plot.png')
    plt.close()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
    ])

    trainset = TeamMateDataset(n_images=50, train=True)
    testset = TeamMateDataset(n_images=10, train=False)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    # Model for CrossEntropyLoss
    model = mobilenet_v3_small(weights=None, num_classes=2).to(device)
    criterion_ce = torch.nn.CrossEntropyLoss()
    print("Training with CrossEntropyLoss:")
    train_and_evaluate(model, criterion_ce, trainloader, testloader, device, "CrossEntropy")

    # Reset model for NLLLoss
    model = mobilenet_v3_small(weights=None, num_classes=2).to(device)
    criterion_nll = torch.nn.NLLLoss()
    print("Training with NLLLoss:")
    train_and_evaluate(model, criterion_nll, trainloader, testloader, device, "NLLLoss")