import theano.tensor as T
import numpy

from theano import function

class HiddenLayer(object):
    """Hidden layer of a neural network"""
    def __init__(self, n_in, n_hid, W = None, b = None, activation = T.nnet.sigmoid):
        super(HiddenLayer, self).__init__()
        self.arg = arg

class Capsule(object):
    """Capsule of a templated autoencoder"""
    def __init__(self, template = None, params = None):
        super(Capsule, self).__init__()
        self.arg = arg


def main():
    train()
    test()

if __name__ == '__main__':
    main()