import torch
from torchvision import datasets, transforms
import utils
from SpreadLoss import SpreadLoss
from CapsNet import capsules
device = torch.device("cpu")


A, B, C, D = 64, 8, 16, 16
model = capsules(A=A, B=B, C=C, D=D, E=10,
                 iters=2).to(device)
model.load_state_dict(torch.load('./model_5.pth'))
model.eval()
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data/mnist', train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1, shuffle=True)


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
            print('\nTest set: Average loss: {:.6f}, Accuracy: {:.6f} \n'.format(
                criterion(output, target, r=1).item(), utils.calculate_accuracy(output, target)[
                    0].item()))
    test_loss /= test_len
    testing_accuracy /= test_len


criterion = SpreadLoss(
    number_of_output_classes=10, m_min=0.2, m_max=0.9)
test(test_loader, model, criterion, device)
