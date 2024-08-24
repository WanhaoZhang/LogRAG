from .mlp import MLP_Autoencoder
from .mlp import MLP


def build_network(net_name):
    """Builds the neural network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'mlp')
    assert net_name in implemented_networks

    net = None

    if net_name == 'mlp':
        net = MLP(768, 128, 32)
    return net


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', "mlp")
    assert net_name in implemented_networks

    ae_net = None

    if net_name == "mlp":
        ae_net = MLP_Autoencoder(768, 128, 32)
        

    return ae_net
