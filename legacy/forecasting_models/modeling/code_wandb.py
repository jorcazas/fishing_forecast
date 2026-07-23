import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import wandb

# Initialize wandb
project_name = "cifar10-itam-example_2"
wandb.init(project=project_name, entity="sergioarnaud")

# setting device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# defining transformations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# defining train and test datasets
trainset = datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

testset = datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# defining our model
model = models.resnet18(pretrained=False)

# change the last layer
model.fc = nn.Linear(model.fc.in_features, 10)

# pass model to device
model = model.to(device)

# Start a new run, tracking hyperparameters in config
config = wandb.config
config.learning_rate = 0.001
config.momentum = 0.9
config.batch_size = 32
config.epochs = 100

# defining loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)

# Watch the model to log model's gradients and parameters
wandb.watch(model, log="all")
class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# start training
for epoch in range(config.epochs): 
    running_loss = 0.0
    total_samples = total_correct = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        correct = (preds == labels).sum().item()
        total_correct += correct
        total_samples += inputs.size(0)

        if i == 0:
            # Log the 0th input, output and prediction to verify the training
            for k,image in enumerate(inputs):
                caption = f'Should be {class_names[labels[k]]} Predicted: {class_names[preds[k]]}'                
                wandb.log(
                    {f"Predicted label {k}": [wandb.Image(image, caption=caption)]}
                )

        running_loss += loss.item()

    # End of epoch, log accuracy and loss for this epoch
    train_acc = total_correct / total_samples
    train_loss = running_loss / len(trainloader)
    wandb.log({"Train Accuracy": train_acc, "Train Loss": train_loss})

    # start evaluation
    total_correct = total_samples = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    val_acc = total_correct / total_samples
    wandb.log({"Validation Accuracy": val_acc})

# save model to wandb
torch.save(model.state_dict(), "model.pth")
wandb.save("model.pth")