import numpy as np
import paddle


class Classifier_Module(paddle.nn.Layer):
    def __init__(self, dims_in, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = paddle.nn.LayerList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                paddle.nn.Conv2d(
                    dims_in,
                    num_classes,
                    kernel_size=3,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=True,
                )
            )
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
            return out


class DeeplabVGG(paddle.nn.Layer):
    def __init__(self, num_classes, vgg16_caffe_path=None, pretrained=False):
        super(DeeplabVGG, self).__init__()
        vgg = paddle.vision.models.vgg16(batch_norm=False)
        if pretrained:
            vgg.set_state_dict(state_dict=paddle.load(path=str(vgg16_caffe_path)))
        features, classifier = list(vgg.features.children()), list(
            vgg.classifier.children()
        )
        features = paddle.nn.Sequential(
            *(features[i] for i in range(23) + range(24, 30))
        )
        for i in [23, 25, 27]:
            features[i].dilation = 2, 2
            features[i].padding = 2, 2
        fc6 = paddle.nn.Conv2d(512, 1024, kernel_size=3, padding=4, dilation=4)
        fc7 = paddle.nn.Conv2d(1024, 1024, kernel_size=3, padding=4, dilation=4)
        self.features = paddle.nn.Sequential(
            *(
                [features[i] for i in range(len(features))]
                + [fc6, paddle.nn.ReLU(), fc7, paddle.nn.ReLU()]
            )
        )
        self.classifier = Classifier_Module(
            1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def optim_parameters(self, args):
        return self.parameters()
