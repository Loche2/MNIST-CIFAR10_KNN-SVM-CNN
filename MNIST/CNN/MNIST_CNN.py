from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm

learning_rate = 0.001
num_epochs = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 定义数据变换
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# 加载MNIST数据集并进行预处理
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


# 定义CNN模型
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # 输入通道数为1，输出通道数为6，卷积核大小为5x5
        self.pool1 = nn.MaxPool2d(2, 2)  # 池化层，窗口大小为2x2
        self.conv2 = nn.Conv2d(6, 16, 5)  # 输入通道数为6，输出通道数为16，卷积核大小为5x5
        self.pool2 = nn.MaxPool2d(2, 2)  # 池化层，窗口大小为2x2
        self.conv3 = nn.Conv2d(16, 120, 5)  # 输入通道数为16，输出通道数为120，卷积核大小为5x5
        self.fc1 = nn.Linear(120, 84)  # 全连接层，输入大小为120，输出大小为84
        self.fc2 = nn.Linear(84, 10)  # 全连接层，输入大小为84，输出大小为10（对应10个类别）

    def forward(self, x):
        x = self.pool1(nn.functional.relu(self.conv1(x)))
        x = self.pool2(nn.functional.relu(self.conv2(x)))
        x = nn.functional.relu(self.conv3(x))
        x = x.view(-1, 120)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 创建模型
model = LeNet5().to(device)
summary(model, (1, 32, 32))

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
losses = []

# 训练网络
print("Start Training!")
for epoch in range(num_epochs):
    loader = tqdm(train_loader, desc=f'Epoch [{epoch + 1}/{num_epochs}]', leave=False)
    total_loss = 0.0  # 用于累积每个epoch的损失值
    for (inputs, labels) in loader:
        model.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loader.set_postfix(loss=loss.item())
    epoch_loss = total_loss / len(train_loader)
    losses.append(epoch_loss)
print("Finished Training!")

# 在测试集上评估模型
correct = 0
total = 0
all_labels = []
all_labels_bin = []
all_predictions = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_labels_bin.extend(label_binarize(labels.cpu().numpy(), classes=np.arange(10)).tolist())
        all_predictions.extend(predicted.cpu().numpy())
        all_probs.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())

print(f"Accuracy on test data: {100 * correct / total}%")

# 绘制混淆矩阵热力图
conf_matrix = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", linewidths=.5, xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# 绘制损失随着epoch变化的折线图
plt.plot(range(num_epochs), losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

# 计算多类别ROC曲线和AUC
fpr = dict()
tpr = dict()
roc_auc = dict()

all_labels_bin = np.array(all_labels_bin)
all_probs = np.array(all_probs)

for i in range(10):
    fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 绘制多类别ROC曲线
plt.figure(figsize=(10, 8))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'purple', 'red', 'green', 'yellow', 'blue', 'orange', 'pink'])
for i, color in zip(range(10), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Multi-class')
plt.legend()
plt.show()
