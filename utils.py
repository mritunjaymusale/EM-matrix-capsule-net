import argparse
import os
import torch
from torchvision import datasets, transforms
from smallNORB import smallNORB


def get_settings():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Matrix-Capsules-EM')
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--em-iters', type=int, default=2, metavar='N',
                        help='iterations of EM Routing')

    parser.add_argument('--save-model-folder', type=str, default='./saved_model', metavar='SF',
                        help='where to store the snapshots')
    parser.add_argument('--data-folder', type=str, default='./data', metavar='DF',
                        help='where to store the datasets')
    parser.add_argument('--dataset', type=str, default='mnist', metavar='D',
                        help='dataset for training(mnist, smallNORB)')
    return parser

def load_mnist(path, args):
    num_class = 10
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(path, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(path, train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True)
    return num_class, train_loader, test_loader

def load_smallNORB(path, args):
    num_class = 5
    train_loader = torch.utils.data.DataLoader(
        smallNORB(path, train=True, download=True,
                  transform=transforms.Compose([
                      transforms.Resize(48),
                      transforms.RandomCrop(32),
                      transforms.ColorJitter(
                          brightness=32./255, contrast=0.5),
                      transforms.ToTensor()
                  ])),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        smallNORB(path, train=False,
                  transform=transforms.Compose([
                      transforms.Resize(48),
                      transforms.CenterCrop(32),
                      transforms.ToTensor()
                  ])),
        batch_size=args.test_batch_size, shuffle=True)
    return num_class, train_loader, test_loader


def load_dataset(args):
    path = os.path.join(args.data_folder, args.dataset)
    if args.dataset == 'mnist':
        num_class, train_loader, test_loader = load_mnist(path, args)
    elif args.dataset == 'smallNORB':
        num_class, train_loader, test_loader = load_smallNORB(path, args)
    else:
        raise NameError('Undefined dataset {}'.format(args.dataset))
    return num_class, train_loader, test_loader





def save_model(model, args):
    path = os.path.join(args.save_model_folder, args.dataset+'.pth')
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    print('saving model to {}'.format(path))
    torch.save(model.state_dict(), path)


def calculate_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_total_trainable_parameters(model):
    
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return pytorch_total_params