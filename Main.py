from load_ORL_faces import FaceDataSet, getTrainValidSamplers
from torch.utils.data import DataLoader

from torchvision import transforms, utils
from ConvNet import Net, Trainer


if __name__ == "__main__":
    dataset = FaceDataSet(root_dir='./data')
    trainSampler, testSampler = getTrainValidSamplers(dataset, validation_split=0.2) 
    trainloader = DataLoader(dataset, sampler=trainSampler, num_workers=1, batch_size=20)
    testloader = DataLoader(dataset, sampler=testSampler, num_workers=1)

    net = Net()
    trainer = Trainer(net, trainloader, testloader, GPU_MODE=True)
    trainer.train(num_epochs=10, visualization=True)
    trainer.predict()
