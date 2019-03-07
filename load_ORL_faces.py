from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


class FaceDataSet(Dataset):
    def __init__(self, root_dir, transform=None):
        data = np.load(root_dir + '/ORL_faces.npz')
        images = np.concatenate((data['trainX'], data['testX']))
        labels = np.concatenate((data['trainY'], data['testY'])).astype(np.int64)
        self.width = 92
        self.height = 112
        self.data = {
            'images': [np.reshape(img, (1, self.height, self.width)) for img in images],
            'labels': [label for label in labels]}
        self.transform = transform

        if (len(self.data['images']) != len(self.data['labels'])):
            raise Exception('len(images) and len(labels) must be equal: len(images) = {0}, len(labels)={1}'.format())

    def __len__(self):
        return len(self.data['images'])

    def __getitem__(self, idx):
        sample = {'image': self.data['images'][idx].astype('float'),
                  'label': self.data['labels'][idx]}
        if self.transform:
            return self.transform(sample)
        else:
            return sample


def getTrainValidSamplers(dataset, validation_split):
    assert validation_split < 1.0 and validation_split > 0.0, 'validation split should be in (0.0,1.0)'
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler
