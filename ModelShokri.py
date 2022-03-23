import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import rescale01
import math
import numpy as np

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class NasrWB(nn.Module):

    def __init__(self, num_classes_in_target, output_size_list, layer_size_list, kernel_size_list):
        super(NasrWB, self).__init__()
        """
        output_size_list: list of integers containing the sizes of the intermediate
        outputs of the target model, that are fed into the "output component".
        layer_size_list:  list of integers containing the output size of FC layers in the
        target model, whose gradient will be fed into the "gradient component".
        kernel_size_list: list of integers containing the input size of FC layers in the
        target model, whose gradient will be fed into the "gradient component".
        """

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Components that analyse the output of the target model and its intermediate layers.
        self.output_component_list = []
        for indx, item in enumerate(output_size_list):
            OutputComp = nn.Sequential(nn.Linear(item, 128),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.2),
                                       nn.Linear(128, 64),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.2))

            if torch.cuda.is_available():
                OutputComp.to(device)
            self.output_component_list.append(OutputComp)

        # Component that analyses the label of the target sample. Input must be in one-hot encoding.
        self.label_component = nn.Sequential(nn.Linear(num_classes_in_target, 128),
                                             nn.ReLU(),
                                             nn.Dropout(p=0.2),
                                             nn.Linear(128, 64),
                                             nn.ReLU(),
                                             nn.Dropout(p=0.2))
        if torch.cuda.is_available():
            self.label_component.to(device)

        # Component that analyses the value of the loss of the model.
        self.loss_component = nn.Sequential(nn.Linear(1, 128),
                                            nn.ReLU(),
                                            nn.Dropout(p=0.2),
                                            nn.Linear(128, 64),
                                            nn.ReLU(),
                                            nn.Dropout(p=0.2))
        if torch.cuda.is_available():
            self.loss_component.to(device)

        # Components that analyse the gradient of the loss w.r.t. the parameters of certain intermediate layers.
        self.gradient_component_list = []
        for indx, kernel_size in enumerate(kernel_size_list):
            num_filters = 4
            GradieComp = nn.Sequential(nn.Conv2d(1, num_filters, (1, kernel_size), stride=1),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.2),
                                       nn.Flatten(),
                                       nn.Linear(num_filters * layer_size_list[indx], 128),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.2),
                                       nn.Linear(128, 64),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.2))
            if torch.cuda.is_available():
                GradieComp.to(device)
            self.gradient_component_list.append(GradieComp)

        # Encoder, combines the outputs of components
        encoder_input_size = 64 * (len(kernel_size_list) + len(output_size_list)) + 128
        self.encoder = nn.Sequential(nn.Linear(encoder_input_size, 256),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.2),
                                     nn.Linear(256, 128),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.2),
                                     nn.Linear(128, 64),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.2),
                                     nn.Linear(64, 1),
                                     nn.Sigmoid())

    def forward(self, inter_outs, loss, label, gradients):
        loss = torch.unsqueeze(loss, 1)
        out1 = self.label_component(label)
        out2 = torch.squeeze(self.loss_component(loss))
        out3_list = []
        for indx, item in enumerate(self.gradient_component_list):
            out3_list.append(item(gradients[indx]))
        out3 = torch.cat(out3_list, 1)
        out4_list = []
        for indx, item in enumerate(self.output_component_list):
            out4_list.append(item(inter_outs[indx]))
        out4 = torch.cat(out4_list, 1)
        return self.encoder(torch.cat((out1, out2, out3, out4), 1))


class DataHandler(Dataset):
    """Handler for training or test data
    """

    def __init__(self, inter0, inter1, loss0, loss1, hot0, hot1, grad0, grad1, indices0, indices1,
                 Max=None, Min=None):

        self.inter0 = inter0
        self.inter1 = inter1
        self.loss0 = loss0
        self.loss1 = loss1
        self.hot0 = hot0
        self.hot1 = hot1
        self.grad0 = grad0
        self.grad1 = grad1
        self.indices0 = indices0
        self.indices1 = indices1

        self.label0 = np.zeros((loss0.shape[0],))
        self.label1 = np.ones((loss1.shape[0],))

        if Max is not None:
            self.Max = Max
        else:
            Max = []

            interMax = []
            for k in range(len(inter0)):
                interMax0 = inter0[k][indices0, :].max(axis=0)
                interMax1 = inter1[k][indices1, :].max(axis=0)
                interMax.append(np.maximum(interMax0, interMax1))
            Max.append(interMax)

            lossMax0 = loss0[indices0].max(axis=0)
            lossMax1 = loss1[indices1].max(axis=0)
            Max.append(np.maximum(lossMax0, lossMax1))

            gradMax = []
            for k in range(len(grad0)):
                gradMax0 = grad0[k][indices0, :, :, :].max(axis=0)
                gradMax1 = grad1[k][indices1, :, :, :].max(axis=0)
                gradMax.append(np.maximum(gradMax0, gradMax1))
            Max.append(gradMax)

            self.Max = Max

        if Min is not None:
            self.Min = Min
        else:
            Min = []

            interMin = []
            for k in range(len(inter0)):
                interMin0 = inter0[k][indices0, :].min(axis=0)
                interMin1 = inter1[k][indices1, :].min(axis=0)
                interMin.append(np.minimum(interMin0, interMin1))
            Min.append(interMin)

            lossMin0 = loss0[indices0].min(axis=0)
            lossMin1 = loss1[indices1].min(axis=0)
            Min.append(np.minimum(lossMin0, lossMin1))

            gradMin = []
            for k in range(len(grad0)):
                gradMin0 = grad0[k][indices0, :, :, :].min(axis=0)
                gradMin1 = grad1[k][indices1, :, :, :].min(axis=0)
                gradMin.append(np.minimum(gradMin0, gradMin1))
            Min.append(gradMin)

            self.Min = Min

    def __len__(self):
        return len(self.indices0) + len(self.indices1)

    def __getitem__(self, i):
        if i < len(self.indices0):
            indx = self.indices0[i]

            inter_i = []
            for k in range(len(self.inter0)):
                inter_aux = self.inter0[k][indx, :]
                inter_aux = rescale01(inter_aux, self.Max[0][k], self.Min[0][k])
                inter_i.append(Tensor(inter_aux))

            loss_i = self.loss0[indx]
            loss_i = rescale01(loss_i, self.Max[1], self.Min[1])
            loss_i = Tensor((loss_i,))

            hot_i = self.hot0[indx]
            hot_i = Tensor(hot_i)

            grad_i = []
            for k in range(len(self.grad0)):
                grad_aux = self.grad0[k][indx, :, :, :]
                grad_aux = rescale01(grad_aux, self.Max[2][k], self.Min[2][k])
                grad_i.append(Tensor(grad_aux))

            label_i = self.label0[indx]
            label_i = Tensor((label_i,))

            if cuda:
                loss_i = loss_i.cuda()
                hot_i = hot_i.cuda()
                for k in range(len(inter_i)):
                    inter_i[k] = inter_i[k].cuda()
                for k in range(len(grad_i)):
                    grad_i[k] = grad_i[k].cuda()
                label_i = label_i.cuda()

            return [inter_i, loss_i, hot_i, grad_i], label_i

        else:
            indx = self.indices1[i - len(self.indices0)]

            inter_i = []
            for k in range(len(self.inter1)):
                inter_aux = self.inter1[k][indx, :]
                inter_aux = rescale01(inter_aux, self.Max[0][k], self.Min[0][k])
                inter_i.append(Tensor(inter_aux))

            loss_i = self.loss1[indx]
            loss_i = rescale01(loss_i, self.Max[1], self.Min[1])
            loss_i = Tensor((loss_i,))

            hot_i = self.hot1[indx]
            hot_i = Tensor(hot_i)

            grad_i = []
            for k in range(len(self.grad1)):
                grad_aux = self.grad1[k][indx, :, :, :]
                grad_aux = rescale01(grad_aux, self.Max[2][k], self.Min[2][k])
                grad_i.append(Tensor(grad_aux))

            label_i = self.label1[indx]
            label_i = Tensor((label_i,))

            if cuda:
                loss_i = loss_i.cuda()
                hot_i = hot_i.cuda()
                for k in range(len(inter_i)):
                    inter_i[k] = inter_i[k].cuda()
                for k in range(len(grad_i)):
                    grad_i[k] = grad_i[k].cuda()
                label_i = label_i.cuda()

            return [inter_i, loss_i, hot_i, grad_i], label_i


def TrainWBAttacker(trainingData, testingData, output_size, layer_size, kernel_size,
                    num_classes_in_target=100, batch_size=64, n_epochs=100, epsilon=1e-3):
    """
    """
    lr = 0.0001

    Len = len(trainingData)
    ValLen = len(testingData)

    dataloader = DataLoader(trainingData, batch_size=batch_size, shuffle=True)

    dataloaderVal = DataLoader(testingData, batch_size=100, shuffle=False)

    AttackModel = NasrWB(num_classes_in_target, output_size, layer_size, kernel_size)
    Loss = nn.BCELoss()

    if cuda:
        AttackModel.cuda()
        Loss.cuda()

    optimizer = torch.optim.Adam(AttackModel.parameters(), lr=lr)
    currLoss = math.inf

    for k in range(n_epochs):  # loop through epochs.
        lostList = []
        Acc = 0
        for i, batch in enumerate(dataloader):  # loop through batches.
            optimizer.zero_grad()

            example = batch[0]
            target = batch[1]

            loss = Loss(AttackModel(*example), target)  # compute the loss.
            loss.backward()  # update the weights.
            optimizer.step()

            aux = sum(torch.eq(torch.ge(AttackModel(*example), 0.5), target))
            Acc = Acc + aux.cpu().data.numpy()

            lostList.append(loss.item())

        # Compute Accuracy over training set.
        Acc = Acc / Len

        # Compute Accuracy over validation set.

        valAcc = 0
        for i, batch in enumerate(dataloaderVal):
            example = batch[0]
            target = batch[1]
            aux = sum(torch.eq(torch.ge(AttackModel(*example), 0.5), target))
            valAcc = valAcc + aux.cpu().data.numpy()
        valAcc = valAcc / ValLen

        prevLoss = currLoss
        currLoss = np.mean(lostList)

        if abs(prevLoss - currLoss) < epsilon:
            break

        print('Loss : %f, Accuracy: %f, Validation Accuracy: %f Iteration: %d' % (currLoss, Acc, valAcc, k + 1))
    return AttackModel
