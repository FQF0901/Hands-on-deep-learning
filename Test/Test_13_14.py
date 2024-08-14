import os
import sys
import torch
import torchvision
from torch import nn
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from d2l import torch as d2l

'''
1. 下载数据集
'''
#@save
# d2l.DATA_HUB['dog_tiny'] = (d2l.DATA_URL + 'kaggle_dog_tiny.zip',
#                             '0cb91d09b814ecdc07b50f31f8dcad3e81d6a86d')

# 如果使用Kaggle比赛的完整数据集，请将下面的变量更改为False
demo = True
if demo:
    data_dir = d2l.download_extract('dog_tiny')
else:
    data_dir = os.path.join('..', 'data', 'dog-breed-identification')

'''
2. 整理数据集
'''
def reorg_dog_data(data_dir, valid_ratio):
    labels = d2l.read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    d2l.reorg_train_valid(data_dir, labels, valid_ratio)
    d2l.reorg_test(data_dir)


batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_dog_data(data_dir, valid_ratio)

'''
3. 图像增广
'''
# torchvision.transforms.Compose用于将多个图像转换操作组合成一个转换管道
# transform_train是一个包含多种转换操作的序列，将应用于训练数据
transform_train = torchvision.transforms.Compose([
    # 在图像上进行随机裁剪，裁剪出的区域会缩放到指定的尺寸（这里是224x224）
    # scale=(0.08, 1.0)定义了裁剪区域相对于原始图像的面积范围，从原始面积的8%到100%之间。
    # ratio=(3.0/4.0, 4.0/3.0)定义了裁剪区域的宽高比范围，从3/4到4/3之间
    torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0)),
    # 以50%的概率对图像进行水平翻转
    torchvision.transforms.RandomHorizontalFlip(),
    # 随机调整图像的亮度、对比度和饱和度
    # brightness=0.4表示亮度会随机在原始值的±40%之间变化
    # contrast=0.4表示对比度会在±40%之间变化
    # saturation=0.4表示饱和度会在±40%之间变化
    torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    # 将PIL图像或NumPy数组转换为PyTorch的Tensor，并将图像像素值从[0, 255]范围缩放到[0, 1]范围
    torchvision.transforms.ToTensor(),
    # 用于标准化图像的每个通道。
    # [0.485, 0.456, 0.406]是每个通道的均值，[0.229, 0.224, 0.225]是每个通道的标准差
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

transform_test = torchvision.transforms.Compose([
    # 将图像的长边调整到256像素，保持图像的纵横比
    torchvision.transforms.Resize(256),
    # 从图像中心裁切224x224大小的图片
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

'''
4. 读取数据集
'''
# torchvision.datasets.ImageFolder用于从指定目录加载图像数据集，并应用预定义的图像转换（transform_train）。
# os.path.join(data_dir, 'train_valid_test', folder)生成每个数据集的目录路径，其中folder是['train', 'train_valid']中的每一个，
# 创建两个数据集train_ds和train_valid_ds
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train_valid_test', folder), transform=transform_train) for folder in ['train', 'train_valid']]

valid_ds, test_ds = [torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train_valid_test', folder), transform=transform_test) for folder in ['valid', 'test']]

# torch.utils.data.DataLoader用于从train_ds和train_valid_ds数据集中创建数据加载器。
# batch_size定义每个批次的样本数，
# shuffle=True表示每个epoch会打乱数据，
# drop_last=True表示如果最后一个批次样本数不足以形成完整批次，则丢弃该批次。
# 结果是train_iter和train_valid_iter两个数据加载器
train_iter, train_valid_iter = [torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, drop_last=True) for dataset in (train_ds, train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False, drop_last=True)

test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False, drop_last=False)

'''
5. 微调预训练模型
'''
def get_net(devices):   # 该函数接受一个参数 devices，这是一个包含PyTorch设备（如CPU或GPU）的列表。函数将使用这些设备来将模型加载到合适的位置上
    finetune_net = nn.Sequential()  # n.Sequential是PyTorch提供的一个模块容器，允许将多个模块按顺序组合在一起
    # 使用torchvision.models.resnet34加载一个预训练的ResNet34模型。
    # pretrained=True表示使用在ImageNet数据集上预训练的权重。将这个模型赋值给 finetune_net.features，作为网络的特征提取部分。
    # 注意，这里的用法不是特别标准，因为 nn.Sequential 本身没有 features 属性，这可能是为了简化模型构造的自定义代码。
    finetune_net.features = torchvision.models.resnet34(pretrained=True)
    # 这里使用 nn.Sequential 创建一个新的输出层 finetune_net.output_new，这个部分包括：
    # nn.Linear(1000, 256)：一个全连接层，将输入的1000维特征映射到256维。
    # nn.ReLU()：一个ReLU激活函数，增加非线性。
    # nn.Linear(256, 120)：另一个全连接层，将256维特征映射到120维，适应120个输出类别的任务（例如分类任务）。
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 120))
    # 将模型参数分配给用于计算的CPU或GPU
    finetune_net = finetune_net.to(devices[0])
    # 遍历 finetune_net.features 部分的所有参数，将 requires_grad 属性设置为 False。这意味着在训练过程中，这些参数不会被更新，只会用于特征提取
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net

loss = nn.CrossEntropyLoss(reduction='none')    # 分类问题，用到概率分布，所以用交叉熵

def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        features, labels = features.to(devices[0]), labels.to(devices[0])
        outputs = net(features)
        l = loss(outputs, labels)
        l_sum += l.sum()
        n += labels.numel()
    return (l_sum / n).to('cpu')

'''
6. 定义训练函数
'''
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay):
    # 只训练小型自定义输出网络
    # 使用 DataParallel 将模型分配到多个设备上，并将模型移动到第一个设备上
    # nn.DataParallel(net, device_ids=devices): 使用 DataParallel 封装模型 net，device_ids 指定了要使用的设备（如 GPU）。DataParallel 会将输入数据分割到多个设备上，并在这些设备上并行计算模型的前向和反向传播。
    # .to(devices[0]): 将 DataParallel 包装后的模型移动到 devices[0] 指定的主设备（通常是第一个 GPU）。虽然数据在多个 GPU 上处理，但模型参数需要存放在主设备上。
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])   
    trainer = torch.optim.SGD((param for param in net.parameters()
                               if param.requires_grad), lr=lr,
                              momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)   # 学习率调度器: 创建一个 StepLR 调度器，以周期性地调整学习率
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            features, labels = features.to(devices[0]), labels.to(devices[0])
            trainer.zero_grad()
            output = net(features)
            l = loss(output, labels).sum()
            l.backward()
            trainer.step()
            metric.add(l, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (metric[0] / metric[1], None))
        measures = f'train loss {metric[0] / metric[1]:.3f}'
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss.detach().cpu()))
        scheduler.step()
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}' f' examples/sec on {str(devices)}')
    
'''
7. 训练和验证模型
'''
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 10, 1e-4, 1e-4
lr_period, lr_decay, net = 2, 0.9, get_net(devices)
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

