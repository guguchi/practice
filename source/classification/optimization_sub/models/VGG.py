from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L


class VGG16(chainer.ChainList):
    def __init__(self, class_labels):
        w = chainer.initializers.HeNormal()
        super(VGG16, self).__init__(
            VGGBlock(64),
            VGGBlock(128),
            VGGBlock(256, 3),
            VGGBlock(512, 3),
            VGGBlock(512, 3, True, class_labels))

    def __call__(self, x):
        for f in self.children():
            x = f(x)
        #if chainer.config.train:
        return x
        #return F.softmax(x)


class VGGBlock(chainer.Chain):
    def __init__(self, n_channels, n_convs=2, fc=False, class_labels=10):
        w = chainer.initializers.HeNormal()
        super(VGGBlock, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, n_channels, 3, 1, 1, initialW=w)
            self.conv2 = L.Convolution2D(
                n_channels, n_channels, 3, 1, 1, initialW=w)
            if n_convs == 3:
                self.conv3 = L.Convolution2D(
                    n_channels, n_channels, 3, 1, 1, initialW=w)
            if fc:
                self.fc4 = L.Linear(None, 4096, initialW=w)
                self.fc5 = L.Linear(4096, 4096, initialW=w)
                self.fc6 = L.Linear(4096, class_labels, initialW=w)

        self.n_convs = n_convs
        self.fc = fc

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        if self.n_convs == 3:
            h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(h, 2, 2)
        if self.fc:
            h = F.dropout(F.relu(self.fc4(h)))
            h = F.dropout(F.relu(self.fc5(h)))
            h = self.fc6(h)
        return h

"""
class Block(chainer.Chain):

    def __init__(self, out_channels, ksize, pad=1):
        super(Block, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(None, out_channels, ksize, pad=pad,
                                        nobias=True)
            self.bn = L.BatchNormalization(out_channels)

    def __call__(self, x):
        h = self.conv(x)
        h = self.bn(h)
        return F.relu(h)


class VGG(chainer.Chain):

    def __init__(self, class_labels=10):
        super(VGG, self).__init__()
        with self.init_scope():
            self.block1_1 = Block(64, 3)
            self.block1_2 = Block(64, 3)
            self.block2_1 = Block(128, 3)
            self.block2_2 = Block(128, 3)
            self.block3_1 = Block(256, 3)
            self.block3_2 = Block(256, 3)
            self.block3_3 = Block(256, 3)
            self.block4_1 = Block(512, 3)
            self.block4_2 = Block(512, 3)
            self.block4_3 = Block(512, 3)
            self.block5_1 = Block(512, 3)
            self.block5_2 = Block(512, 3)
            self.block5_3 = Block(512, 3)
            self.fc1 = L.Linear(None, 512, nobias=True)
            self.bn_fc1 = L.BatchNormalization(512)
            self.fc2 = L.Linear(None, class_labels, nobias=True)

    def __call__(self, x):
        # 64 channel blocks:
        h = self.block1_1(x)
        h = F.dropout(h, ratio=0.3)
        h = self.block1_2(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 128 channel blocks:
        h = self.block2_1(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block2_2(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 256 channel blocks:
        h = self.block3_1(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block3_2(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block3_3(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 512 channel blocks:
        h = self.block4_1(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block4_2(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block4_3(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 512 channel blocks:
        h = self.block5_1(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block5_2(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block5_3(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.dropout(h, ratio=0.5)
        h = self.fc1(h)
        h = self.bn_fc1(h)
        h = F.relu(h)
        h = F.dropout(h, ratio=0.5)
        return self.fc2(h)
"""
