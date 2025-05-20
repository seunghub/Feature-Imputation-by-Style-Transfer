import torch

class FeatureExtractor(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super(FeatureExtractor, self).__init__()

        layers = []
        layers += [torch.nn.Linear(in_size, in_size), torch.nn.ReLU(), torch.nn.Dropout(0.1)]
        layers += [torch.nn.Linear(in_size, in_size), torch.nn.ReLU(), torch.nn.Dropout(0.1)]
        layers += [torch.nn.Linear(in_size, out_size)]
        self.laysers = torch.nn.Sequential(*layers)

    def forward(self, A0):
        x = self.laysers(A0)
        return x

class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class Classifier(torch.nn.Module):
    def __init__(self, in_size, out_size, domain_clf=False):
        super(Classifier, self).__init__()
        self.domain_clf = domain_clf
        layers = []
        layers += [torch.nn.Linear(in_size, in_size), torch.nn.ReLU(), torch.nn.Dropout(0.1)]
        layers += [torch.nn.Linear(in_size, out_size)]
        self.laysers = torch.nn.Sequential(*layers)

    def forward(self, embd, alpha=None):
        if self.domain_clf:
            embd = ReverseLayerF.apply(embd, alpha)
        x = self.laysers(embd)
        return x

class Generator(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super(Generator, self).__init__()
        layers = []
        layers += [torch.nn.Linear(in_size, in_size), torch.nn.ReLU(), torch.nn.Dropout(0.1)]
        layers += [torch.nn.Linear(in_size, in_size), torch.nn.ReLU(), torch.nn.Dropout(0.1)]
        layers += [torch.nn.Linear(in_size, out_size)]
        self.laysers = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.laysers(x)
        return x

class Discriminator(torch.nn.Module):
    def __init__(self, in_size):
        super(Discriminator, self).__init__()
        layers = []
        layers += [torch.nn.Linear(in_size, in_size), torch.nn.ReLU(), torch.nn.Dropout(0.1)]
        layers += [torch.nn.Linear(in_size, in_size), torch.nn.ReLU(), torch.nn.Dropout(0.1)]
        layers += [torch.nn.Linear(in_size, 1)]
        layers += [torch.nn.Sigmoid()]
        self.laysers = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.laysers(x)
        return x