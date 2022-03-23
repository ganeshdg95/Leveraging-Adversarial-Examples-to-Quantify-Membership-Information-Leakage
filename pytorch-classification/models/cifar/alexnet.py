'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
import torch.nn as nn


__all__ = ['alexnet']


class AlexNet(nn.Module):

    def __init__(self, Max, Min, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)
        self.Max = Max
        self.Min = Min
        
    def unscale(self,tensor,max_,min_):
        max_ = max_.reshape(1,-1,1,1)
        min_ = min_.reshape(1,-1,1,1)
        return tensor*(max_-min_)+min_
    
    def intermediate_forward(self, x, layer_index):
        x = self.unscale(x,self.Max,self.Min)
        x1 = self.features[0](x)
        x1 = self.features[1](x1)
        x1 = self.features[2](x1)        
        x1 = self.features[3](x1)
        x1 = self.features[4](x1)
        if layer_index == 0:
            return x1
        
        x2 = self.features[5](x1)       
        x2 = self.features[6](x2)
        x2 = self.features[7](x2)
        if layer_index == 1:
            return x2
        
        x3 = self.features[8](x2)
        x3 = self.features[9](x3)
        if layer_index == 2:
            return x3
        
        x4 = self.features[10](x3)
        x4 = self.features[11](x4)
        if layer_index == 3:
            return x4
        
        x5 = self.features[12](x4)        
        x5 = x5.view(x5.size(0), -1)       
        x5 = self.classifier(x5)
        if layer_index == 4:
            return x5
        
    def list_forward(self, x):
        x = self.unscale(x,self.Max,self.Min)
        x1 = self.features[0](x)
        x1 = self.features[1](x1)
        x1 = self.features[2](x1)        
        x1 = self.features[3](x1)
        x1 = self.features[4](x1)
        
        x2 = self.features[5](x1)       
        x2 = self.features[6](x2)
        x2 = self.features[7](x2)
        
        x3 = self.features[8](x2)
        x3 = self.features[9](x3)
        
        x4 = self.features[10](x3)
        x4 = self.features[11](x4)
        
        x5 = self.features[12](x4)        
        x5 = x5.view(x5.size(0), -1)       
        x5 = self.classifier(x5)
        
        return [x1,x2,x3,x4,x5]

    def forward(self, x):
        x = self.unscale(x,self.Max,self.Min)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def alexnet(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet(**kwargs)
    return model
