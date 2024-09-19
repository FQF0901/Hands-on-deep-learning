import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

# 定义数据集
class CustomDataset(Dataset):
    def __init__(self, transform=None):
        self.data = []  # 用于存储图片和标签
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, target = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, target

# 定义Transformer网络
class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc1 = nn.Linear(128 * 56 * 96, 512)  # 根据输出形状调整
        self.transformer = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6)
        self.fc_out = nn.Linear(512, 60)  # 输出30个点的坐标 (x, y)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc1(x).unsqueeze(0)  # 添加batch维度
        x = self.transformer(x, x)
        x = self.fc_out(x)
        return x.view(-1, 30, 2)  # (batch_size, 30, 2)

# 超参数
learning_rate = 0.001
num_epochs = 20
batch_size = 16

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 384)),
])

# 创建数据集和数据加载器
dataset = CustomDataset(transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型、损失函数和优化器
model = TransformerModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练流程
for epoch in range(num_epochs):
    for images, targets in data_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
