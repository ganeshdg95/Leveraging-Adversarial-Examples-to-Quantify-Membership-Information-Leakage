from autoattack import AutoAttack
import torch
from torch.autograd import grad
import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc
from sklearn.metrics import roc_curve, accuracy_score

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def rescale(tensor, max_, min_):
    max_ = max_.reshape(1, -1, 1, 1)
    min_ = min_.reshape(1, -1, 1, 1)
    return (tensor - min_) / (max_ - min_ + 1e-8)


def unscale(tensor, max_, min_):
    max_ = max_.reshape(1, -1, 1, 1)
    min_ = min_.reshape(1, -1, 1, 1)
    return tensor * (max_ - min_) + min_


def predict(model, sample):
    _, pred = torch.max(model(sample), 1)
    return pred


def to_one_hot(y, num_classes):
    if len(y.shape) == 1:
        y = torch.unsqueeze(y, 1)
    y_one_hot = torch.zeros(y.shape[0], num_classes)
    if cuda:
        y_one_hot = y_one_hot.cuda()
    y_one_hot = y_one_hot.scatter(1, y, 1)
    return y_one_hot


def randSplitDF(scores, seed, test_size, train_set=False, train_size=None):
    """Split input dataframe into random subsets for testing and training"""
    np.random.seed(seed)
    sets = []

    scores = scores.sample(frac=1).reset_index(drop=True)

    evaluation_set = scores.iloc[:test_size, :]
    sets.append(evaluation_set)

    if train_set and (train_size is not None):
        training_set = scores.iloc[test_size:test_size + train_size, :]
        sets.append(training_set)

    return sets


def randSplit(scores, seed, test_size, train_set=False, train_size=None):
    """Split input dataframe into random subsets for testing and training"""
    np.random.seed(seed)
    sets = []

    indexes = np.arange(scores.shape[0])
    np.random.shuffle(indexes)
    scores = scores[indexes, :]

    evaluation_set = scores[:test_size, :]
    sets.append(evaluation_set)

    if train_set and (train_size is not None):
        training_set = scores[test_size:test_size + train_size, :]
        sets.append(training_set)

    return sets


def computeMetrics(scores0, scores1, FPR):
    labels0 = np.zeros_like(scores0)
    labels1 = np.ones_like(scores1)

    scores = np.concatenate((scores0, scores1))
    labels = np.concatenate((labels0, labels1))

    # ROC curve
    fpr, tpr, thr = roc_curve(labels, scores)
    TPR = np.interp(FPR, fpr, tpr)

    # FPR @TPR95

    metrics = np.interp([.80, .85, .90, .95], tpr, fpr)

    # AUROC

    AUROC = roc_auc(labels, scores)

    # Optimal Accuracy

    AccList = [accuracy_score(scores > t, labels) for t in thr]
    Acc_opt = np.max(AccList)

    metrics = np.append((AUROC, Acc_opt), metrics)

    return TPR, metrics


def computeMetricsAlt(scores, labels, FPR):
    # ROC curve
    fpr, tpr, thr = roc_curve(labels, scores)
    TPR = np.interp(FPR, fpr, tpr)

    # FPR @TPR95

    metrics = np.interp([.80, .85, .90, .95], tpr, fpr)

    # AUROC

    AUROC = roc_auc(labels, scores)

    # Optimal Accuracy

    AccList = [accuracy_score(scores > t, labels) for t in thr]
    Acc_opt = np.max(AccList)

    metrics = np.append((AUROC, Acc_opt), metrics)

    return TPR, metrics


def computeBestThreshold(scores0, scores1):
    labels0 = np.zeros_like(scores0)
    labels1 = np.ones_like(scores1)

    scores = np.concatenate((scores0, scores1))
    labels = np.concatenate((labels0, labels1))

    # ROC curve
    _, _, thr = roc_curve(labels, scores)

    AccList = [accuracy_score(scores0 > t, labels0) + accuracy_score(scores1 > t, labels1) for t in thr]
    Acc_opt_indx = np.argmax(AccList)

    return thr[Acc_opt_indx]


def evalBestThreshold(thr_opt, scores0, scores1):
    labels0 = np.zeros_like(scores0)
    labels1 = np.ones_like(scores1)

    Acc = (accuracy_score(scores0 > thr_opt, labels0) + accuracy_score(scores1 > thr_opt, labels1)) / 2

    FPR = sum(scores0 > thr_opt) / len(scores0)

    return [Acc, FPR[0]]


def evalThresholdAlt(threshold, scores, labels):
    scores0 = scores[(labels == 0)]
    scores1 = scores[(labels == 1)]
    labels0 = labels[(labels == 0)]
    labels1 = labels[(labels == 1)]

    Acc = (accuracy_score(scores0 > threshold, labels0) + accuracy_score(scores1 > threshold, labels1)) / 2

    FPR = sum(scores0 > threshold) / len(scores0)

    return [Acc, FPR]


def Softmax(in_tensor):
    """ Apply Softmax to the input tensor. 
    in_tensor: pytorch tensor.
    """
    in_tensor = torch.exp(in_tensor)
    sum_ = torch.unsqueeze(torch.sum(in_tensor, 1), 1)
    return torch.div(in_tensor, sum_)


def softmaxAttack(model, data):
    """ Produces the scores for the Softmax attack.
    model: target model. Must be an instance of a pytorch model.
    data: samples to be tested. Pytorch tensor with shape: [batch,channels,...].
    """
    scores, _ = torch.max(Softmax(model(data).detach()), 1)
    return scores


def advDistance(model, images, labels, batch_size=10, epsilon=1, norm='Linf'):
    if norm == 'Linf':
        ordr = float('inf')
    elif norm == 'L1':
        ordr = 1
    elif norm == 'L2':
        ordr = 2

    adversary = AutoAttack(model, norm=norm, eps=epsilon, version='standard')
    adv = adversary.run_standard_evaluation(images, labels, bs=batch_size)

    dist = Dist(images, adv, ordr=ordr)

    return dist


def gradNorm(model, images, labels, Loss):
    loss = Loss(model(images), labels)
    gNorm = []
    for j in range(loss.shape[0]):
        grad_ = grad(loss[j], model.parameters(), create_graph=True)
        gNorm_ = -torch.sqrt(sum([grd.norm() ** 2 for grd in grad_]))
        gNorm.append(gNorm_.detach())
    return torch.stack(gNorm)


def lossAttack(model, images, labels, Loss):
    loss = Loss(model(images).detach(), labels)
    return -loss


def Dist(sample, adv, ordr=float('inf')):
    sus = sample - adv
    sus = sus.view(sus.shape[0], -1)
    return torch.norm(sus, ordr, 1)


def rescale01(data, Max, Min):
    """ Rescale input features to [0,1]. 
    data: pytorch tensor of shape [batch,features].
    Max: Tensor with shape [1].
    Min: Tensor with shape [1].
    """
    return (data - Min) / (Max - Min + 1e-8)


def Entropy(softprob):
    epsilon = 1e-8
    return - torch.sum(softprob * torch.log(softprob + epsilon), 1)


def ModEntropy(softprob, labels):
    epsilon = 1e-8
    confidence = torch.stack([softprob[i, labels[i]] for i in range(softprob.shape[0])])
    firstTerm = (confidence - 1) * torch.log(confidence + epsilon)
    secondTerm = - torch.sum(softprob * torch.log(1 - softprob + epsilon), 1)
    excessTerm = confidence * torch.log(1 - confidence + epsilon)
    return firstTerm + secondTerm + excessTerm
