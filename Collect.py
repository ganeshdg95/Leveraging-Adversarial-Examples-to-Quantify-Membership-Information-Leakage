import argparse
import random
from tqdm import tqdm as tq
import sys
from collections import OrderedDict
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import grad
import numpy as np
import pandas as pd
from utils import rescale, Softmax, advDistance, to_one_hot, Entropy, ModEntropy
from pathlib import Path
import importlib
resnext = importlib.import_module("pytorch-classification.models.cifar.resnext")
resnet = importlib.import_module("pytorch-classification.models.cifar.resnet")
alexnet = importlib.import_module("pytorch-classification.models.cifar.alexnet")
densenet = importlib.import_module("pytorch-classification.models.cifar.densenet")

parser = argparse.ArgumentParser(description='Apply different strategies for MIA to target model.')

parser.add_argument('--seed', type=int, help='Set random seed for reproducibility.')
parser.add_argument('--dataset', type=str, default='cifar100', help='Which dataset to use for the experiments.')
parser.add_argument('--model_type', type=str, help='Model Architecture to attack.')
parser.add_argument('--batch_size', type=int, default=50, help='Batch size for batched computations.')
parser.add_argument('--output_dir', type=str, default='./', help='Where to store output data.')
parser.add_argument('--data_dir', type=str, default='./data', help='Where to retrieve the dataset.')
parser.add_argument('--trained_dir', type=str, default='./trained_models', help='Where to retrieve trained models.')
parser.add_argument('--dry_run', action='store_true', default=False, help='Test run on 100 samples.')

exp_parameters = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Record time of computation

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()

# Setting seed for reproducibility

seed = exp_parameters.seed

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Loading Datasets

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if exp_parameters.dataset == 'cifar100':
    dataloader = datasets.CIFAR100
    num_classes = 100
elif exp_parameters.dataset == 'cifar10':
    dataloader = datasets.CIFAR10
    num_classes = 10

batch_size = exp_parameters.batch_size

data_dir = exp_parameters.data_dir
train_dataset = dataloader(data_dir, train=True, download=True, transform=transform_train)
test_dataset = dataloader(data_dir, train=False, download=True, transform=transform_test)

train_set = DataLoader(train_dataset, batch_size=len(train_dataset) if not exp_parameters.dry_run else 100)
images1 = next(iter(train_set))[0]
labels1 = next(iter(train_set))[1]

test_set = DataLoader(test_dataset, batch_size=len(test_dataset) if not exp_parameters.dry_run else 100)
images0 = next(iter(test_set))[0]
labels0 = next(iter(test_set))[1]

# Preprocessing data

if cuda:
    images1 = images1.cuda()
    images0 = images0.cuda()
    labels1 = labels1.cuda()
    labels0 = labels0.cuda()

num_channels = 3

max1, _ = torch.max(images1.transpose(0, 1).reshape(num_channels, -1), 1)
min1, _ = torch.min(images1.transpose(0, 1).reshape(num_channels, -1), 1)

max0, _ = torch.max(images0.transpose(0, 1).reshape(num_channels, -1), 1)
min0, _ = torch.min(images0.transpose(0, 1).reshape(num_channels, -1), 1)

Max = torch.max(max0, max1)
Min = torch.min(min0, min1)

images1 = rescale(images1, Max, Min)
images0 = rescale(images0, Max, Min)

# Loading model

model_dir = exp_parameters.trained_dir
if exp_parameters.dataset == 'cifar10':
    model_dir = model_dir + '/cifar10'
model_type = exp_parameters.model_type

if model_type == 'DenseNet':
    num_layers = 5
    compute_grad = [567]
    model = densenet.DenseNet(Max, Min, depth=190, num_classes=num_classes, growthRate=40)
    checkpoint = torch.load(model_dir + '/densenet-bc-L190-k40/model_best.pth.tar', map_location=torch.device('cpu'))
elif model_type == 'ResNext':
    num_layers = 4
    compute_grad = [93]
    model = resnext.CifarResNeXt(8, 29, num_classes, Max, Min)
    checkpoint = torch.load(model_dir + '/resnext-8x64d/model_best.pth.tar', map_location=torch.device('cpu'))
elif model_type == 'ResNet':
    num_layers = 4
    compute_grad = [498]
    model = resnet.ResNet(164, Max, Min, num_classes=num_classes, block_name='bottleneck')
    checkpoint = torch.load(model_dir + '/resnet-110/model_best.pth.tar', map_location=torch.device('cpu'))
elif model_type == 'AlexNet':
    num_layers = 5
    compute_grad = [10]
    model = alexnet.AlexNet(Max, Min, num_classes=num_classes)
    checkpoint = torch.load(model_dir + '/alexnet/model_best.pth.tar', map_location=torch.device('cpu'))

Loss = torch.nn.CrossEntropyLoss(reduction='none')

if cuda:
    model.cuda()
    Loss.cuda()

state_dict = checkpoint["state_dict"]
new_state_dict = OrderedDict()
counter = 1
for k, v in state_dict.items():
    if exp_parameters.dataset == 'cifar10':
        if counter < len(state_dict) - 1:
            name = k[:9] + k[16:]
        else:
            name = k
        counter += 1
    else:
        name = k[7:]
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model.eval()

# Results will be save in a csv file 

scoreLists0 = []
scoreLists1 = []

TrainData4Nasr0 = []
TrainData4Nasr1 = []

AddInfoAttacker = []

# Intermediate Outputs computation

with torch.no_grad():
    output_size_list = []

    fwrList0 = []
    for i in range(0, images0.shape[0], batch_size):
        outList = model.list_forward(images0[i:i + batch_size])
        auxList = []
        for j in range(len(outList)):
            out = outList[j].detach()
            out = out.view(out.shape[0], out.shape[1], -1)
            out = torch.mean(out, 2)
            auxList.append(out)
        fwrList0.append(auxList[:])

    fwrList0_ = []
    for j in range(len(fwrList0[0])):
        auxList = []
        for i in range(len(fwrList0)):
            auxList.append(fwrList0[i][j].detach())
        interOuts = torch.cat(auxList, 0)
        print(interOuts.shape)
        output_size_list.append(interOuts.shape[1])
        fwrList0_.append(interOuts)

    fwrList1 = []
    for i in range(0, images1.shape[0], batch_size):
        outList = model.list_forward(images1[i:i + batch_size])
        auxList = []
        for j in range(len(outList)):
            out = outList[j].detach()
            out = out.view(out.shape[0], out.shape[1], -1)
            out = torch.mean(out, 2)
            auxList.append(out)
        fwrList1.append(auxList[:])

    fwrList1_ = []
    for j in range(len(fwrList1[0])):
        auxList = []
        for i in range(len(fwrList1)):
            auxList.append(fwrList1[i][j].detach())
        interOuts = torch.cat(auxList, 0)
        print(interOuts.shape)
        fwrList1_.append(interOuts)

print('Intermediate outputs computation: done')

sys.stdout.flush()
sys.stderr.flush()

for layer in fwrList0_:
    TrainData4Nasr0.append(layer.cpu().data.numpy())
for layer in fwrList1_:
    TrainData4Nasr1.append(layer.cpu().data.numpy())

AddInfoAttacker.append(np.asarray(output_size_list))

# Softmax attack

scores0, _ = torch.max(Softmax(fwrList0_[-1]), 1)
scores1, _ = torch.max(Softmax(fwrList1_[-1]), 1)

print('Softmax Attack computation: done')

scores0 = scores0.cpu().data.numpy()
scores1 = scores1.cpu().data.numpy()

scoreLists0.append(scores0)
scoreLists1.append(scores1)

# Entropy Computation

softprob0 = Softmax(fwrList0_[-1]).detach()
softprob1 = Softmax(fwrList1_[-1]).detach()

Entr0 = -Entropy(softprob0).cpu().data.numpy()
Entr1 = -Entropy(softprob1).cpu().data.numpy()

print('Entropy computation: done')

scoreLists0.append(Entr0)
scoreLists1.append(Entr1)

# Modified Entropy 

ModEntr0 = -ModEntropy(softprob0, labels0).cpu().data.numpy()
ModEntr1 = -ModEntropy(softprob1, labels1).cpu().data.numpy()

print('Modified Entropy computation: done')

sys.stdout.flush()
sys.stderr.flush()

scoreLists0.append(ModEntr0)
scoreLists1.append(ModEntr1)

# Adversarial Distance attack Linf

LinfDist0 = advDistance(model, images0, labels0, batch_size=batch_size, epsilon=1, norm='Linf')
LinfDist1 = advDistance(model, images1, labels1, batch_size=batch_size, epsilon=1, norm='Linf')

print('Adversarial Distance Linf computation: done')

sys.stdout.flush()
sys.stderr.flush()

scoreLists0.append(LinfDist0.cpu().data.numpy())
scoreLists1.append(LinfDist1.cpu().data.numpy())

# L2

L2Dist0 = advDistance(model, images0, labels0, batch_size=batch_size, norm='L2', epsilon=images0.shape[2])
L2Dist1 = advDistance(model, images1, labels1, batch_size=batch_size, norm='L2', epsilon=images1.shape[2])

print('Adversarial Distance L2 computation: done')

sys.stdout.flush()
sys.stderr.flush()

scoreLists0.append(L2Dist0.cpu().data.numpy())
scoreLists1.append(L2Dist1.cpu().data.numpy())

# L1

L1Dist0 = advDistance(model, images0, labels0, batch_size=batch_size, norm='L1', epsilon=images0.shape[2] ** 2)
L1Dist1 = advDistance(model, images1, labels1, batch_size=batch_size, norm='L1', epsilon=images1.shape[2] ** 2)

print('Adversarial Distance L1 computation: done')

sys.stdout.flush()
sys.stderr.flush()

scoreLists0.append(L1Dist0.cpu().data.numpy())
scoreLists1.append(L1Dist1.cpu().data.numpy())

# Loss attack
with torch.no_grad():
    loss0_ = []
    for i in range(0, images0.shape[0], batch_size):
        loss0_.append(Loss(model(images0[i:i + batch_size]), labels0[i:i + batch_size]).detach())
    loss0 = torch.cat(loss0_, 0)

    loss1_ = []
    for i in range(0, images1.shape[0], batch_size):
        loss1_.append(Loss(model(images1[i:i + batch_size]), labels1[i:i + batch_size]).detach())
    loss1 = torch.cat(loss1_, 0)

print('Loss computation: done')

sys.stdout.flush()
sys.stderr.flush()

scoreLists0.append(-loss0.cpu().data.numpy())
scoreLists1.append(-loss1.cpu().data.numpy())

TrainData4Nasr0.append(loss0.cpu().data.numpy())
TrainData4Nasr1.append(loss1.cpu().data.numpy())

# Labels in one hot encoding

labels0_1hot = to_one_hot(labels0, num_classes)
labels1_1hot = to_one_hot(labels1, num_classes)

TrainData4Nasr0.append(labels0_1hot.cpu().data.numpy())
TrainData4Nasr1.append(labels1_1hot.cpu().data.numpy())

# Gradient computation for Nasr Attacker and Rezaei attacker

gradients0 = []
gradients1 = []
layer_size_list = []
kernel_size_list = []
one_sample = torch.unsqueeze(images0[0], 0)
loss_one_sample = Loss(model(one_sample), torch.unsqueeze(labels0[0], 0))
grad_ = grad(loss_one_sample[0], model.parameters(), create_graph=True)
for j in compute_grad:
    gradients0.append(torch.zeros((images0.shape[0], grad_[j].shape[0], grad_[j].shape[1])))
    gradients1.append(torch.zeros((images1.shape[0], grad_[j].shape[0], grad_[j].shape[1])))
    layer_size_list.append(grad_[j].shape[0])
    kernel_size_list.append(grad_[j].shape[1])

grad0L1 = []
grad0L2 = []
grad0Linf = []
grad0mean = []
grad0Skew = []
grad0Kurt = []
grad0AbsMin = []

for j in tq(range(images0.shape[0])):
    one_sample = torch.unsqueeze(images0[j], 0)
    loss_one_sample = Loss(model(one_sample), torch.unsqueeze(labels0[j], 0))
    grad_ = grad(loss_one_sample[0], model.parameters(), create_graph=True)
    for i, l in enumerate(compute_grad):
        gradients0[i][j] = grad_[l].detach()
    aux_grad_list = []
    for i in range(len(grad_)):
        aux_grad_list.append(torch.flatten(grad_[i]).detach())
    grad_ = torch.cat(aux_grad_list, 0).detach()

    # Statistics on the gradient
    Gmean = torch.mean(grad_)
    Gdiffs = grad_ - Gmean
    Gvar = torch.mean(torch.pow(Gdiffs, 2.0))
    Gstd = torch.pow(Gvar, 0.5)
    Gzscores = Gdiffs / Gstd
    Gskews = torch.mean(torch.pow(Gzscores, 3.0))
    Gkurtoses = torch.mean(torch.pow(Gzscores, 4.0)) - 3.0

    GL1 = torch.norm(grad_, 1)
    GL2 = torch.norm(grad_, 2)
    GLinf = torch.norm(grad_, float('inf'))
    GabsMin = torch.min(torch.abs(grad_))

    grad0L1.append(GL1)
    grad0L2.append(GL2)
    grad0Linf.append(GLinf)
    grad0mean.append(Gmean)
    grad0Skew.append(Gskews)
    grad0Kurt.append(Gkurtoses)
    grad0AbsMin.append(GabsMin)

    sys.stdout.flush()
    sys.stderr.flush()

grad0L1 = torch.stack(grad0L1).cpu().data.numpy()
grad0L2 = torch.stack(grad0L2).cpu().data.numpy()
grad0Linf = torch.stack(grad0Linf).cpu().data.numpy()
grad0mean = torch.stack(grad0mean).cpu().data.numpy()
grad0Skew = torch.stack(grad0Skew).cpu().data.numpy()
grad0Kurt = torch.stack(grad0Kurt).cpu().data.numpy()
grad0AbsMin = torch.stack(grad0AbsMin).cpu().data.numpy()

grad1L1 = []
grad1L2 = []
grad1Linf = []
grad1mean = []
grad1Skew = []
grad1Kurt = []
grad1AbsMin = []

for j in tq(range(images1.shape[0])):
    one_sample = torch.unsqueeze(images1[j], 0)
    loss_one_sample = Loss(model(one_sample), torch.unsqueeze(labels1[j], 0))
    grad_ = grad(loss_one_sample[0], model.parameters(), create_graph=True)
    for i, l in enumerate(compute_grad):
        gradients1[i][j] = grad_[l].detach()
    aux_grad_list = []
    for i in range(len(grad_)):
        aux_grad_list.append(torch.flatten(grad_[i]).detach())
    grad_ = torch.cat(aux_grad_list, 0).detach()

    # Statistics on the gradient
    Gmean = torch.mean(grad_)
    Gdiffs = grad_ - Gmean
    Gvar = torch.mean(torch.pow(Gdiffs, 2.0))
    Gstd = torch.pow(Gvar, 0.5)
    Gzscores = Gdiffs / Gstd
    Gskews = torch.mean(torch.pow(Gzscores, 3.0))
    Gkurtoses = torch.mean(torch.pow(Gzscores, 4.0)) - 3.0

    GL1 = torch.norm(grad_, 1)
    GL2 = torch.norm(grad_, 2)
    GLinf = torch.norm(grad_, float('inf'))
    GabsMin = torch.min(torch.abs(grad_))

    grad1L1.append(GL1)
    grad1L2.append(GL2)
    grad1Linf.append(GLinf)
    grad1mean.append(Gmean)
    grad1Skew.append(Gskews)
    grad1Kurt.append(Gkurtoses)
    grad1AbsMin.append(GabsMin)

    sys.stdout.flush()
    sys.stderr.flush()

grad1L1 = torch.stack(grad1L1).cpu().data.numpy()
grad1L2 = torch.stack(grad1L2).cpu().data.numpy()
grad1Linf = torch.stack(grad1Linf).cpu().data.numpy()
grad1mean = torch.stack(grad1mean).cpu().data.numpy()
grad1Skew = torch.stack(grad1Skew).cpu().data.numpy()
grad1Kurt = torch.stack(grad1Kurt).cpu().data.numpy()
grad1AbsMin = torch.stack(grad1AbsMin).cpu().data.numpy()

print('Gradient computation: done')

# Gradients and additional info for Nasr attacker

for i in range(len(gradients0)):
    gradients0[i] = torch.unsqueeze(gradients0[i], 1)
    gradients1[i] = torch.unsqueeze(gradients1[i], 1)

for layer in gradients0:
    TrainData4Nasr0.append(layer.cpu().data.numpy())
for layer in gradients1:
    TrainData4Nasr1.append(layer.cpu().data.numpy())

AddInfoAttacker.append(np.asarray(layer_size_list))
AddInfoAttacker.append(np.asarray(kernel_size_list))

# Gradient Norms

scoreLists0.append(-grad0L1)
scoreLists0.append(-grad0L2)
scoreLists0.append(-grad0Linf)
scoreLists0.append(-grad0mean)
scoreLists0.append(-grad0Skew)
scoreLists0.append(grad0Kurt)
scoreLists0.append(grad0AbsMin)

scoreLists1.append(-grad1L1)
scoreLists1.append(-grad1L2)
scoreLists1.append(-grad1Linf)
scoreLists1.append(-grad1mean)
scoreLists1.append(-grad1Skew)
scoreLists1.append(grad1Kurt)
scoreLists1.append(grad1AbsMin)

# Gradient computation w.r.t. model input

grad0L1 = []
grad0L2 = []
grad0Linf = []
grad0mean = []
grad0Skew = []
grad0Kurt = []
grad0AbsMin = []

for j in tq(range(images0.shape[0])):
    one_sample = torch.unsqueeze(images0[j], 0)
    one_sample.requires_grad = True
    loss_one_sample = Loss(model(one_sample), torch.unsqueeze(labels0[j], 0))
    grad_ = grad(loss_one_sample[0], one_sample, create_graph=True)
    grad_ = torch.flatten(grad_[0]).detach()

    # Statistics on the gradient
    Gmean = torch.mean(grad_)
    Gdiffs = grad_ - Gmean
    Gvar = torch.mean(torch.pow(Gdiffs, 2.0))
    Gstd = torch.pow(Gvar, 0.5)
    Gzscores = Gdiffs / Gstd
    Gskews = torch.mean(torch.pow(Gzscores, 3.0))
    Gkurtoses = torch.mean(torch.pow(Gzscores, 4.0)) - 3.0

    GL1 = torch.norm(grad_, 1)
    GL2 = torch.norm(grad_, 2)
    GLinf = torch.norm(grad_, float('inf'))
    GabsMin = torch.min(torch.abs(grad_))

    grad0L1.append(GL1)
    grad0L2.append(GL2)
    grad0Linf.append(GLinf)
    grad0mean.append(Gmean)
    grad0Skew.append(Gskews)
    grad0Kurt.append(Gkurtoses)
    grad0AbsMin.append(GabsMin)

    sys.stdout.flush()
    sys.stderr.flush()

grad0L1 = torch.stack(grad0L1).cpu().data.numpy()
grad0L2 = torch.stack(grad0L2).cpu().data.numpy()
grad0Linf = torch.stack(grad0Linf).cpu().data.numpy()
grad0mean = torch.stack(grad0mean).cpu().data.numpy()
grad0Skew = torch.stack(grad0Skew).cpu().data.numpy()
grad0Kurt = torch.stack(grad0Kurt).cpu().data.numpy()
grad0AbsMin = torch.stack(grad0AbsMin).cpu().data.numpy()

grad1L1 = []
grad1L2 = []
grad1Linf = []
grad1mean = []
grad1Skew = []
grad1Kurt = []
grad1AbsMin = []

for j in tq(range(images1.shape[0])):
    one_sample = torch.unsqueeze(images1[j], 0)
    one_sample.requires_grad = True
    loss_one_sample = Loss(model(one_sample), torch.unsqueeze(labels1[j], 0))
    grad_ = grad(loss_one_sample[0], one_sample, create_graph=True)
    grad_ = torch.flatten(grad_[0]).detach()

    # Statistics on the gradient
    Gmean = torch.mean(grad_)
    Gdiffs = grad_ - Gmean
    Gvar = torch.mean(torch.pow(Gdiffs, 2.0))
    Gstd = torch.pow(Gvar, 0.5)
    Gzscores = Gdiffs / Gstd
    Gskews = torch.mean(torch.pow(Gzscores, 3.0))
    Gkurtoses = torch.mean(torch.pow(Gzscores, 4.0)) - 3.0

    GL1 = torch.norm(grad_, 1)
    GL2 = torch.norm(grad_, 2)
    GLinf = torch.norm(grad_, float('inf'))
    GabsMin = torch.min(torch.abs(grad_))

    grad1L1.append(GL1)
    grad1L2.append(GL2)
    grad1Linf.append(GLinf)
    grad1mean.append(Gmean)
    grad1Skew.append(Gskews)
    grad1Kurt.append(Gkurtoses)
    grad1AbsMin.append(GabsMin)

    sys.stdout.flush()
    sys.stderr.flush()

grad1L1 = torch.stack(grad1L1).cpu().data.numpy()
grad1L2 = torch.stack(grad1L2).cpu().data.numpy()
grad1Linf = torch.stack(grad1Linf).cpu().data.numpy()
grad1mean = torch.stack(grad1mean).cpu().data.numpy()
grad1Skew = torch.stack(grad1Skew).cpu().data.numpy()
grad1Kurt = torch.stack(grad1Kurt).cpu().data.numpy()
grad1AbsMin = torch.stack(grad1AbsMin).cpu().data.numpy()

print('Gradient computation w.r.t. model input: done')

scoreLists0.append(-grad0L1)
scoreLists0.append(-grad0L2)
scoreLists0.append(-grad0Linf)
scoreLists0.append(-grad0mean)
scoreLists0.append(grad0Skew)
scoreLists0.append(grad0Kurt)
scoreLists0.append(-grad0AbsMin)

scoreLists1.append(-grad1L1)
scoreLists1.append(-grad1L2)
scoreLists1.append(-grad1Linf)
scoreLists1.append(-grad1mean)
scoreLists1.append(grad1Skew)
scoreLists1.append(grad1Kurt)
scoreLists1.append(-grad1AbsMin)

# Reporting total time of computation

end.record()
torch.cuda.synchronize()

print('Elapsed time of computation in miliseconds: %f' % (start.elapsed_time(end)))

# Saving Results

scoreLists0 = np.transpose(np.stack(scoreLists0))

df_scores0 = pd.DataFrame(scoreLists0, columns=['Softmax Response', 'Entropy', 'Modified Entropy',
                                                'Adversarial Distance Linf',
                                                'Adversarial Distance L2',
                                                'Adversarial Distance L1',
                                                'Loss Value',
                                                'Grad wrt model parameters L1',
                                                'Grad wrt model parameters L2',
                                                'Grad wrt model parameters Linf',
                                                'Grad wrt model parameters Mean',
                                                'Grad wrt model parameters Skewness',
                                                'Grad wrt model parameters Kurtosis',
                                                'Grad wrt model parameters Abs Min',
                                                'Grad wrt input image L1',
                                                'Grad wrt input image L2',
                                                'Grad wrt input image Linf',
                                                'Grad wrt input image Mean',
                                                'Grad wrt input image Skewness',
                                                'Grad wrt input image Kurtosis',
                                                'Grad wrt input image Abs Min'])

scoreLists1 = np.transpose(np.stack(scoreLists1))

df_scores1 = pd.DataFrame(scoreLists1, columns=['Softmax Response', 'Entropy', 'Modified Entropy',
                                                'Adversarial Distance Linf',
                                                'Adversarial Distance L2',
                                                'Adversarial Distance L1',
                                                'Loss Value',
                                                'Grad wrt model parameters L1',
                                                'Grad wrt model parameters L2',
                                                'Grad wrt model parameters Linf',
                                                'Grad wrt model parameters Mean',
                                                'Grad wrt model parameters Skewness',
                                                'Grad wrt model parameters Kurtosis',
                                                'Grad wrt model parameters Abs Min',
                                                'Grad wrt input image L1',
                                                'Grad wrt input image L2',
                                                'Grad wrt input image Linf',
                                                'Grad wrt input image Mean',
                                                'Grad wrt input image Skewness',
                                                'Grad wrt input image Kurtosis',
                                                'Grad wrt input image Abs Min'])

outdir = Path(exp_parameters.output_dir)
results_dir = outdir / 'RawResults'
results_dir.mkdir(parents=True, exist_ok=True)

df_scores0.to_csv(results_dir / f'scores0_{model_type}_partial.csv', index=False)
df_scores1.to_csv(results_dir / f'scores1_{model_type}_partial.csv', index=False)

np.savez_compressed(results_dir / f'NasrTrain0_{model_type}.npz', *TrainData4Nasr0)
np.savez_compressed(results_dir / f'NasrTrain1_{model_type}.npz', *TrainData4Nasr1)
np.savez_compressed(results_dir / f'NasrAddInfo_{model_type}.npz', *AddInfoAttacker)
