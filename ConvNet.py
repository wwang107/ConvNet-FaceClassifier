# Neural Network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 25 * 20, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 25 * 20)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Trainer(object):
    def __init__(self, Net, trainloader, testlaoder, GPU_MODE=False):
        self.device = self.getGPU()
        self.net = Net.to(self.device)
        self.trainloader = trainloader
        self.testloader = testlaoder
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(),
                                   lr=0.001, momentum=0.9)
        self.GPU_MODE = GPU_MODE

    def train(self, num_epochs, visualization=False):
        print("start training\n")
        running_loss = 0.0
        for epoch in range(0, num_epochs):
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs
                inputs, labels = data['image'], data['label']
                inputs, labels = inputs.float().to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 10 == 0:    # print every 10 mini-batches
                    print('[%d, %5d] loss: %.3f' % (
                        epoch + 1, i + 1, running_loss))
                    running_loss = 0.0
                    self.predict(visualization=visualization)

        print('Finished Trainning')

    def predict(self, visualization=False):
        correct = 0
        total = 0
        vis_images = []
        vis_labels = []
        vis_preditions = []
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data['image'], data['label']
                images, labels = images.float().to(self.device), labels.to(self.device)
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if visualization:
                    vis_images.append(images)
                    vis_labels.append(labels)
                    vis_preditions.append(predicted)
        print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))

        if visualization:
            self.visualizePrediction(vis_images, vis_preditions, vis_labels)
    
    def visualizePrediction(self, images, y_hat, y):
        # import ipdb;ipdb.set_trace()
        num_predictions = len(y_hat)
        if num_predictions > 4:
            num_predictions = 4
        # start plot predition and ground truth
        fig = plt.figure()
        for i in range(0, num_predictions):
            npimg = images[i].cpu().numpy()
            ax = plt.subplot(2, num_predictions, i + 1)
            plt.tight_layout()
            ax.set_title('prediction #{}'.format(y_hat[i].item()), color='b')
            ax.axis('off')
            
            plt.imshow(npimg.reshape((112, 92)), cmap='gray')

            ax = plt.subplot(2, num_predictions, num_predictions + i + 1)
            plt.tight_layout()
            ax.set_title('groundtruth #{}'.format(y[i].item()))
            ax.axis('off')
            plt.imshow(npimg.reshape((112, 92)), cmap='gray')
        plt.show()
        plt.close(fig)

    def getGPU(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        return device
