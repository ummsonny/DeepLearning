import torch
import torchvision
from torchvision.transforms import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optimizer
import torch.optim.lr_scheduler as lr_scheduler
import time
import torch.nn.functional as F
# from Args import *
import wandb
from torch.utils.data.sampler import SubsetRandomSampler
import torchsummary

# 모든 데이터를 이용해서 채널별로 평균과 표준편차를 구한다.
transform = transforms.Compose([
    transforms.ToTensor()
])
dataset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=False, transform=transform)
mean = dataset.data.mean(axis=(0, 1, 2))  # 원본 데이터를 가지고 논다.
std = dataset.data.std(axis=(0, 1, 2))
# print(mean, std)  # 0~255
# print(dataset[0])  # 0~1


Args = {"batch_size": 64,
        "mean": mean / 255,
        "std": std / 255,
        "name": "VGG16_5"
        }

transforms_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomRotation(degrees=20), # 데이터 증강1(선택 가능) https://www.analyticsvidhya.com/blog/2021/04/10-pytorch-transformations-you-need-to-know/
    transforms.RandomHorizontalFlip(), # 데이터 증강2
    transforms.RandomRotation(degrees=20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(Args["mean"], Args["std"])  # 데이터 정규화 http://www.gisdeveloper.co.kr/?p=8168
])

transforms_valid = transforms.Compose([
    transforms.Resize((32, 32)),  # 데이터 리사이즈
    transforms.ToTensor(),
    transforms.Normalize(Args["mean"], Args["std"])  # 데이터 정규화 http://www.gisdeveloper.co.kr/?p=8168
])

transforms_test = transforms.Compose([
    transforms.Resize((32, 32)),  # 데이터 리사이즈
    transforms.ToTensor(),
    transforms.Normalize(Args["mean"], Args["std"])  # 데이터 정규화 http://www.gisdeveloper.co.kr/?p=8168
])

batch_size = Args["batch_size"]

# 여기 확인!
train_set = torchvision.datasets.CIFAR10(root='drive/MyDrive/cifar10', train=True, transform=transforms_train,
                                         download=False)
valid_set = torchvision.datasets.CIFAR10(root='drive/MyDrive/cifar10', train=True, transform=transforms_valid,
                                         download=False)
test_set = torchvision.datasets.CIFAR10(root='drive/MyDrive/cifar10', train=False, transform=transforms_test,
                                        download=False)

# 이미지 셋 갯수 확인하기
num_train = len(train_set)  # 50000
num_test = len(test_set)  # 10000
val_size = 0.3

# val 데이터 만들기
index = list(range(num_train))
np.random.shuffle(index)

split = int(np.floor(val_size * num_train))
val_idx, train_idx = index[:split], index[split:]

train_samp = SubsetRandomSampler(train_idx)
val_samp = SubsetRandomSampler(val_idx)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_samp, num_workers=0)
val_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, sampler=val_samp, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)

dataloaders = {'train': train_loader, 'valid': val_loader, 'test': test_loader}
dataset_sizes = {'train': len(train_loader.dataset), 'valid': len(val_loader.dataset), 'test': len(test_loader.dataset)}

print(dataset_sizes['train'], '----', dataset_sizes['valid'], '-------', dataset_sizes['test'])

# 데이터셋의 클래스 종류 구하기
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  # 여기 cifar10데이터셋의 인덱스가 곧 레이블이다. 사슴은 4, 개구리는 6이 정답 레이블이다.

# classes = train_set.classes
# print(train_set.class_to_idx)
# print(classes)
# print(len(classes))


# 불러온 이미지 확인

def image_show(img):
    npimg = img.numpy()
    npimg = npimg * 0.5 + 0.5  # unnormalize

    plt.imshow(np.transpose(npimg, (1, 2, 0)))


print(train_set[0][0].shape)
print(train_set[3][1])
# image_show(train_set[3][0])


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.vgg_name = vgg_name  # 원하는 vgg
        self.features = self.make_layers(cfg[vgg_name])  # vgg 네트워크 중간부분 구현하기
        self.classifier = nn.Sequential(  # 마지막 분류기 구현
            nn.Linear(512, 4096),  # 여기 고쳐야할듯
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        out = self.features(x)  # vgg 네트워크 중간부분 통과
        # print(out.shape,'***************')
        out = out.view(out.size(0), -1)  # 텐서 평탄화하기
        out = self.classifier(out)
        #return F.log_softmax(out, dim=1)
        return out

    def make_layers(self, cfg):  # vgg 네트워크 중간부분 구현
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=x, kernel_size=(3, 3), stride=1, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)  # inplace 메모리 감소
                           ]
                in_channels = x
        return nn.Sequential(*layers).apply(self.weight_init_He_uniform)

    def weight_init_He_uniform(self, submodule):
        if isinstance(submodule, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(submodule.weight)
        elif isinstance(submodule, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(submodule.weight)

# GPU 사용을 위한 device 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch.optim.lr_scheduler as lr_scheduler

# 학습을 위한 설정(finetuning)
model = VGG('VGG16')
#model = models.vgg16(pretrained=True)
if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
epochs = 100
optimizer_ft = optimizer.Adam(model.parameters(), lr=0.001)  # https://gomguard.tistory.com/187
Scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7,
                                gamma=0.1)  # 7에폭마다 최적화시 학습률(lr)를 0.1를 곱해 감소시킨다. https://gaussian37.github.io/dl-pytorch-lr_scheduler/

wandb.login
wandb.init(project="cifar10_VGG16", name=Args["name"], reinit=True)
wandb.config.update(Args)
wandb.watch(model)
print(model)
torchsummary.summary(model, input_size=(3,32,32))

# wandb 사용법 검색
# https://greeksharifa.github.io/references/2020/06/10/wandb-usage/

def Train(optimizer_ft):
    running_loss = 0.0
    running_corrects = 0
    total = 0
    optimizer = optimizer_ft

    # 각 Epoch은 학습 단계와 검증 단계를 거침
    model.train()
    for batch_idx, (inputs, labels) in enumerate(dataloaders['train']):
        # print(batch_idx)
        if torch.cuda.is_available():
            inputs = inputs.to(device)
            labels = labels.to(device)
        else:
            inputs, labels = inputs, labels

        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        total += labels.size(0)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_corrects += (preds == labels).sum().item()

        np_train_loss = np.average(running_loss)
        np_train_acc = np.average(100 * running_corrects / total)

    epoch_loss = np_train_loss / dataset_sizes['train']  # 한 에폭의 손실함수값 평균
    epoch_acc = np_train_acc  # 한 에폭의 정확도 평균

    print('{} Loss: {:.4f} Acc: {:.4f}'.format('train', epoch_loss, epoch_acc))

    wandb.log({
        "Train Loss": epoch_loss,
        "Train Accuracy": epoch_acc,
        "Train error": 100 - epoch_acc,
        "lr": optimizer.param_groups[0]['lr'],  # 학습률 로깅
    })


def Val():
    with torch.no_grad():

        v_running_loss = 0.0
        v_running_corrects = 0.0
        v_total = 0

        # 각 Epoch은 학습 단계와 검증 단계를 거침
        model.eval()
        for batch_idx, (val_inputs, val_labels) in enumerate(dataloaders['valid']):
            # print(batch_idx)
            if torch.cuda.is_available():
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)
            else:
                val_inputs, val_labels = val_inputs, val_labels

                # optimizer.zero_grad()

            outputs = model(val_inputs)
            _, v_preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, val_labels)

            v_total += val_labels.size(0)

            v_running_loss += loss.item()
            v_running_corrects += (v_preds == val_labels).sum().item()

            np_val_loss = np.average(v_running_loss)
            np_val_acc = np.average(100 * v_running_corrects / v_total)

        val_epoch_loss = np_val_loss / dataset_sizes['valid']  # 한 에폭의 손실함수값 평균
        val_epoch_acc = np_val_acc  # 한 에폭의 정확도 평균

        print('{} Loss: {:.4f} Acc: {:.4f}'.format('valid', val_epoch_loss, val_epoch_acc))

        Scheduler.step()

        wandb.log({
            "Valid Loss": val_epoch_loss,
            "Valid Accuracy": val_epoch_acc,
            "Valid error": 100 - val_epoch_acc,
        })
    return val_epoch_acc


# train함수 정의
def train(model, criterion, optimizer, scheduler, num_epochs=5):  # 내가 설정한 여러 에폭을 돌린다.

    since = time.time()

    # model.state_dict()
    best_model_wts = model.state_dict()  # 신경망 각 레이어의 가중치들을 담아놓는다.
    best_acc = 0.0

    for epoch in range(num_epochs):  # 반복문 1바퀴가 1에폭이다.
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        Train(optimizer_ft)

        epoch_acc = Val()

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))


# 이미지넷으로 pretrained된 model의 가중치를 그대로 가져와 학습
# FC 레이어 층의 채널만을 수정하여 학습
model_ft = train(model, criterion, optimizer_ft, lr_scheduler, epochs)

# 스크래치 학습(기존 가중치없이 처음부터 학습)
# model_ft2 = train2(model, criterion, optimizer_ft, lr_scheduler, epochs)
