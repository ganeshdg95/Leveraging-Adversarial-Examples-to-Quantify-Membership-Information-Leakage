from autoattack import AutoAttack
import torch
from torch.autograd import grad
import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc
from sklearn.metrics import roc_curve, accuracy_score

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def rescale(tensor, max_, min_):
    """ Rescale a pytorch tensor to [0,1].
    tensor: pytorch tensor with dimensions [batch,channels,width,height]
    max_: pytorch tensor containing the maximum values per channel
    min_: pytorch tensor containing the minimum values per channel
    outputs -> rescaled pytorch tensor with the same dimensions as 'tensor'
    """
    max_ = max_.reshape(1, -1, 1, 1)
    min_ = min_.reshape(1, -1, 1, 1)
    return (tensor - min_) / (max_ - min_ + 1e-8)


def unscale(tensor, max_, min_):
    """ Rescale a pytorch tensor back to its original values.
    tensor: pytorch tensor with dimensions [batch,channels,width,height]
    max_: pytorch tensor containing the maximum values per channel
    min_: pytorch tensor containing the minimum values per channel
    outputs -> rescaled pytorch tensor with the same dimensions as 'tensor'
    """
    max_ = max_.reshape(1, -1, 1, 1)
    min_ = min_.reshape(1, -1, 1, 1)
    return tensor * (max_ - min_) + min_


def predict(model, sample):
    """ returns the index of the predicted class.
    model: instance of a nn.Module subclass
    sample: pytorch tensor. Batch of samples. Must have the appropriate dimensions for an input of 'model'
    outputs -> tensor containing the index of the predicted classes for 'sample'
    """
    _, pred = torch.max(model(sample), 1)
    return pred


def to_one_hot(y, num_classes):
    """ Convert a list of indexes into a list of one-hot encoded vectors.
    y: pytorch tensor of shape [batch], containing a list of integer labels
    num_classes: int corresponding to the number of classes
    outputs -> pytorch tensor containing the one-hot encoding of the labels 'y'. Its dimensions are [batch,num_classes]
    """
    if len(y.shape) == 1:
        y = torch.unsqueeze(y, 1)
    y_one_hot = torch.zeros(y.shape[0], num_classes)
    if cuda:
        y_one_hot = y_one_hot.cuda()
    y_one_hot = y_one_hot.scatter(1, y, 1)
    return y_one_hot


def computeMetrics(scores0, scores1, FPR):
    """ Computes performance metrics using the scores computed with a certain strategy. The performance scores computed
    include the AUROC score, the best accuracy achieved for any threshold and the FPR at TPR 95%. Also computes the TPR
    values corresponding to the FPR values given as input.
    scores0: numpy array containing the negative scores
    scores1: numpy array containing the positive scores
    FPR: numpy array. TPR values will be interpolated for this FPR values
    outputs -> tuple of ('TPR', 'metrics'). 'TPR' is a numpy array. 'metrics' is a list containing the AUROC score, best
    accuracy, FPR at TPR 80, 85, 90 and 95%
    """
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


def computeBestThreshold(scores0, scores1):
    """ Computes the threshold which maximizes the accuracy given the scores.
        scores0: numpy array containing the negative scores
        scores1: numpy array containing the positive scores
        outputs -> thresh achieving the best accuracy for the input scores. Float
        """
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
    """ Computes the balanced accuracy and FPR of the given scores with the given threshold.
    thr_opt: float threshold
    scores0: numpy array containing the negative scores
    scores1: numpy array containing the positive scores
    outputs -> tuple containing the balanced accuracy and FPR

    """
    labels0 = np.zeros_like(scores0)
    labels1 = np.ones_like(scores1)

    Acc = (accuracy_score(scores0 > thr_opt, labels0) + accuracy_score(scores1 > thr_opt, labels1)) / 2

    FPR = sum(scores0 > thr_opt) / len(scores0)

    return [Acc, FPR[0]]


def Softmax(in_tensor):
    """ Apply Softmax to the input tensor. 
    in_tensor: pytorch tensor with dimensions [batch, length]
    outputs -> pytorch tensor with the same dimensions as 'in_tensor' containing the softmax of 'in_tensor'
    """
    in_tensor = torch.exp(in_tensor)
    sum_ = torch.unsqueeze(torch.sum(in_tensor, 1), 1)
    return torch.div(in_tensor, sum_)


def softmaxAttack(model, data):
    """ Produces the scores for the Softmax attack, which is the maximum value of the softmax vector for each sample
    model: instance of a nn.Module subclass
    data: samples to be tested. Pytorch tensor with shape appropriate shape for 'model'
    outputs -> pytorch tensor of dimensions [batch] containing the softmax score of the input
    """
    scores, _ = torch.max(Softmax(model(data).detach()), 1)
    return scores


def advDistance(model, images, labels, batch_size=10, epsilon=1, norm='Linf'):
    """ Computes the adversarial distance score. First, adversarial examples are computed for each sample that is
    correctly classified by the target model. Then, the distance between the original and adversarial samples is
    computed. If a sample is misclassified, resulting adversarial distance will be 0.
    model: instance of a nn.Module subclass
    images: pytorch tensor with dimensions [batch,channels,width,height]
    labels: pytorch tensor of shape [batch] containing the integer labels of the 'images'
    batch_size: integer indicating the batch size for computing adversarial examples
    epsilon: maximum value for the magnitude of perturbations
    norm: indicates the norm used for computing adversarial examples and for measuring the distance between samples.
    Must be in {'Linf','L2','L1'}
    outputs -> pytorch tensor of dimensions [batch] containing the adversarial distance of 'images'
    """
    if norm == 'Linf':
        ordr = float('inf')
    elif norm == 'L1':
        ordr = 1
    elif norm == 'L2':
        ordr = 2

    if cuda:
        dev = 'cuda'
    else:
        dev = 'cpu'

    adversary = AutoAttack(model, norm=norm, eps=epsilon, version='standard', device=dev)
    adv = adversary.run_standard_evaluation(images, labels, bs=batch_size)

    dist = Dist(images, adv, ordr=ordr)

    return dist


def gradNorm(model, images, labels, Loss):
    """ Computes the l2 norm of the gradient of the loss w.r.t. the model parameters
    model: instance of a nn.Module subclass
    images: pytorch tensor with dimensions [batch,channels,width,height]
    labels: pytorch tensor of shape [batch] containing the integer labels of the samples
    loss: callable, loss function
    outputs -> pytorch tensor of dimensions [batch] containing the l2 norm of the gradients
    """
    loss = Loss(model(images), labels)
    gNorm = []
    for j in range(loss.shape[0]): # Loop over the batch
        grad_ = grad(loss[j], model.parameters(), create_graph=True)
        gNorm_ = -torch.sqrt(sum([grd.norm() ** 2 for grd in grad_]))
        gNorm.append(gNorm_.detach())
    return torch.stack(gNorm)


def lossAttack(model, images, labels, Loss):
    """ Computes the loss value for a batch of samples.
    model: instance of a nn.Module subclass
    images: pytorch tensor with dimensions [batch,channels,width,height]
    labels: pytorch tensor of shape [batch] containing the integer labels of the samples
    loss: callable, loss function
    outputs -> pytorch tensor of dimensions [batch] containing the negative loss values
    """
    loss = Loss(model(images).detach(), labels)
    return -loss


def Dist(sample, adv, ordr=float('inf')):
    """Computes the norm of the difference between two vectors. The operation is done for batches of vectors
    sample: pytorch tensor with dimensions [batch, others]
    adv: pytorch tensor with the same dimensions as 'sample'
    ordr: order of the norm. Must be in {1,2,float('inf')}
    outputs -> pytorch tensor of dimensions [batch] containing distance values for the batch of samples.
    """
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
    """ Compute the Shannon Entropy of a vector of soft probabilities.
    softprob: pytorch tensor. Vector of soft probabilities with shape [batch,num_classes]
    outputs -> pytorch tensor containing the entropy of each sample in the batch
    """
    epsilon = 1e-8
    return - torch.sum(softprob * torch.log(softprob + epsilon), 1)


def ModEntropy(softprob, labels):
    """Compute the modified entropy, described https://www.usenix.org/system/files/sec21fall-song.pdf, of a vector of
    soft probabilities.
    softprob: pytorch tensor. Vector of soft probabilities with shape [batch,num_classes]
    labels: pytorch tensor of shape [batch] containing the integer labels of the samples
    outputs -> pytorch tensor containing the modified entropy of each sample in the batch
    """
    epsilon = 1e-8
    confidence = torch.stack([softprob[i, labels[i]] for i in range(softprob.shape[0])])
    firstTerm = (confidence - 1) * torch.log(confidence + epsilon)
    secondTerm = - torch.sum(softprob * torch.log(1 - softprob + epsilon), 1)
    excessTerm = confidence * torch.log(1 - confidence + epsilon)
    return firstTerm + secondTerm + excessTerm
