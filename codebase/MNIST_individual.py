import torch.utils.data as data
class MNIST_individual(data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        img, target = self.data[index], self.label[index]
        return img, target

    def __len__(self):
        return len(self.data)
