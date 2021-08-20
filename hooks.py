import torch.nn as nn

visualization = {}

def get_activation(name):
    def hook_fn(model, input, output):
        visualization[name] = output.cpu().detach()
    return hook_fn

def get_all_layers(net):
    for name, layer in net._modules.items():
        if isinstance(layer, nn.Sequential):
            get_all_layers(layer)
        else:
            if name in ["conv1", "conv5"]:
                layer.register_forward_hook(get_activation(name))