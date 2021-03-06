import torch
import torch.nn as nn
import torch.nn.init as init


class MaskBaseModel(nn.Module):
    def __init__(self, num_classes=3, pretrained=False, freeze=False):
        super().__init__()

        self.net = self.build_layers(num_classes, pretrained)

        self._initialize_weights(pretrained)

        if freeze:
            self._freeze_net()

    def build_layers(self, num_classes, pretrained):
        raise NotImplementedError

    def classifier_layers(self):
        """ returns classifier layers to be referenced for weight freezing or weight initializing """

        raise NotImplementedError

    def forward(self, x):
        return self.net(x)

    def _initialize_weights(self, pretrained):
        """ initialize all weights if not pretrained, otherwise initialize only classifier weights """
        modules = self.modules() if not pretrained else self.classifier_layers()
        for m in modules:
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _freeze_net(self):
        """ freeze all layers excepting classifier layers """
        for param in self.net.parameters():
            param.requires_grad = False

        for param in self.classifier_layers():
            param.requires_grad = True


class AlexNet(MaskBaseModel):

    def build_layers(self, num_classes, pretrained):
        from torchvision.models import alexnet
        net = alexnet(pretrained=pretrained)
        net.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        return net

    def classifier_layers(self):
        """ returns classifier layers to be referenced for weight freezing or lr targeting """

        return self.net.classifier


class VGG19(MaskBaseModel):

    def build_layers(self, num_classes, pretrained):
        from torchvision.models import vgg19_bn
        net = vgg19_bn(pretrained=pretrained)
        net.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        return net

    def classifier_layers(self):
        """ returns classifier layers to be referenced for weight freezing or lr targeting """

        return self.net.classifier


class Resnet18(MaskBaseModel):

    def build_layers(self, num_classes, pretrained):
        from torchvision.models import resnet18
        net = resnet18(pretrained=pretrained)
        net.fc = nn.Linear(512, num_classes)
        return net

    def classifier_layers(self):
        """ returns classifier layers to be referenced for weight freezing or lr targeting """

        return nn.Sequential(self.net.fc)
