from __future__ import print_function
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from CapsNet import capsules

from SpreadLoss import SpreadLoss
import utils


def train(train_loader, model, criterion, optimizer, epoch, device):

    model.train()
    train_len = len(train_loader)
    training_accuracy = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        r = (1.*batch_idx + (epoch-1)*train_len) / (args.epochs*train_len)
        loss = criterion(output, target, r)
        accuracy = utils.calculate_accuracy(output, target)
        loss.backward()
        optimizer.step()

        training_accuracy += accuracy[0].item()
        if batch_idx % 10 == 0:
            print('Train Epoch: {}\t[{}/{} ({:.0f}%)]\t'
                  'Loss: {:.6f}\tAccuracy: {:.6f}\t'.format(
                      epoch, batch_idx * len(data), len(train_loader.dataset),
                      100. * batch_idx / len(train_loader),
                      loss.item(), accuracy[0].item()))
    return training_accuracy


def test(test_loader, model, criterion, device):
    model.eval()
    test_loss = 0
    testing_accuracy = 0
    test_len = len(test_loader)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target, r=1).item()
            testing_accuracy += utils.calculate_accuracy(output, target)[
                0].item()

    test_loss /= test_len
    testing_accuracy /= test_len
    print('\nTest set: Average loss: {:.6f}, Accuracy: {:.6f} \n'.format(
        test_loss, testing_accuracy))
    return testing_accuracy


def main():
    global args

    # Training settings
    parser = utils.get_settings()

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(1337)
    if args.cuda:
        torch.cuda.manual_seed(1337)

    device = torch.device("cuda" if args.cuda else "cpu")

    # datasets
    number_of_output_classes, training_dataset, testing_dataset = utils.load_dataset(
        args)

    # architecture size
    A, B, C, D = 64, 8, 16, 16
    # A, B, C, D = 32, 32, 32, 32
    model = capsules(A=A, B=B, C=C, D=D, E=number_of_output_classes,
                     iters=args.em_iters).to(device)

    criterion = SpreadLoss(
        number_of_output_classes=number_of_output_classes, m_min=0.2, m_max=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=1)

    best_accuracy = test(testing_dataset, model, criterion, device)
    for epoch in range(1, args.epochs + 1):
        accuracy = train(training_dataset, model, criterion,
                         optimizer, epoch, device)
        accuracy /= len(training_dataset)
        scheduler.step(accuracy)
        best_accuracy = max(best_accuracy, test(
            testing_dataset, model, criterion, device))
    best_accuracy = max(best_accuracy, test(
        testing_dataset, model, criterion, device))
    print('best test accuracy: {:.6f}'.format(best_accuracy))

    utils.save_model(model, args)


if __name__ == '__main__':
    main()
