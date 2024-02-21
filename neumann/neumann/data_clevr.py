import os

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image


def load_images_and_labels(dataset='clevr-hans3', split='train', pos_ratio=1.0, neg_ratio=1.0, base=None):
    """Load image paths and labels for clevr-hans dataset.
    """
    image_paths = []
    labels = []
    folder = 'data/clevr-hans/' + dataset + '/' + split + '/'
    true_folder = folder + 'true/'
    false_folder = folder + 'false/'

    filenames = sorted(os.listdir(true_folder))
    n = int(pos_ratio * len(filenames))
    for filename in filenames[:n]:
        if filename != '.DS_Store':
            image_paths.append(os.path.join(true_folder, filename))
            labels.append(1)

    filenames = sorted(os.listdir(false_folder))
    n = int(neg_ratio * len(filenames))
    for filename in filenames[:n]:
        if filename != '.DS_Store':
            image_paths.append(os.path.join(false_folder, filename))
            labels.append(0)
    return image_paths, labels


def load_image_clevr(path):
    """Load an image using given path.
    """
    img = cv2.imread(path)  # BGR
    assert img is not None, 'Image Not Found ' + path
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and HWC to CHW
    img = np.ascontiguousarray(img)
    return img



class CLEVRHans(torch.utils.data.Dataset):
    """CLEVRHans dataset.
    The implementations is mainly from https://github.com/ml-research/NeSyConceptLearner/blob/main/src/pretrain-slot-attention/data.py.
    """

    def __init__(self, dataset, split,  pos_ratio=1.0, neg_ratio=1.0, img_size=128, base=None):
        super().__init__()
        self.img_size = img_size
        self.dataset = dataset
        assert split in {
            "train",
            "val",
            "test",
        }  # note: test isn't very useful since it doesn't have ground-truth scene information
        self.split = split
        self.transform = transforms.Compose(
            [transforms.Resize((img_size, img_size))]
        )
        self.image_paths, self.labels = load_images_and_labels(
            dataset=dataset, split=split, pos_ratio=pos_ratio, neg_ratio=neg_ratio)

    def __getitem__(self, item):
        path = self.image_paths[item]
        image = Image.open(path).convert("RGB")
        image = transforms.ToTensor()(image)[:3, :, :]
        image = self.transform(image)
        image = (image - 0.5) * 2.0  # Rescale to [-1, 1].
        label = torch.tensor(self.labels[item], dtype=torch.float32)
        return image.unsqueeze(0), label


    def __old__getitem__(self, item):
        path = self.image_paths[item]
        image = Image.open(path).convert("RGB")
        image = transforms.ToTensor()(image)[:3, :, :]
        image = self.transform(image)
        image = (image - 0.5) * 2.0  # Rescale to [-1, 1].
        if self.dataset == 'clevr-hans3':
            labels = torch.zeros((3, ), dtype=torch.float32)
        elif self.dataset == 'clevr-hans7':
            labels = torch.zeros((7, ), dtype=torch.float32)
        labels[self.labels[item]] = 1.0
        return image, labels

    def __len__(self):
        return len(self.labels)

class CLEVRConcept(torch.utils.data.Dataset):
    """The Concept-learning dataset for CLEVR-Hans.
    """

    def __init__(self, dataset, split):
        self.dataset = dataset
        self.split = split
        self.data, self.labels = self.load_csv()
        print('concept data: ', self.data.shape, 'labels: ', len(self.labels))

    def load_csv(self):
        data = []
        labels = []
        pos_csv_data = pd.read_csv(
            'data/clevr/concept_data/' + self.split + '/' + self.dataset + '_pos' + '.csv', delimiter=' ')
        pos_data = pos_csv_data.values
        #pos_labels = np.ones((len(pos_data, )))
        pos_labels = np.zeros((len(pos_data, )))
        neg_csv_data = pd.read_csv(
            'data/clevr/concept_data/' + self.split + '/' + self.dataset + '_neg' + '.csv', delimiter=' ')
        neg_data = neg_csv_data.values
        #neg_labels = np.zeros((len(neg_data, )))
        neg_labels = np.ones((len(neg_data, )))
        data = torch.tensor(np.concatenate(
            [pos_data, neg_data], axis=0), dtype=torch.float32)
        labels = torch.tensor(np.concatenate(
            [pos_labels, neg_labels], axis=0), dtype=torch.float32)
        return data, labels

    def __getitem__(self, item):
        return self.data[item], self.labels[item]

    def __len__(self):
        return len(self.data)
