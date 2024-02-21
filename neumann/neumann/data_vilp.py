import os

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image


def load_images_and_labels(dataset='member', split='train', base=None, pos_ratio=1.0, neg_ratio=1.0):
    """Load image paths and labels for clevr-hans dataset.
    """
    image_paths_list = []
    labels = []
    base_folder = 'data/vilp/' + dataset + '/' + split + '/true/'
    folder_names = sorted(os.listdir(base_folder))
    if '.DS_Store' in folder_names:
        folder_names.remove('.DS_Store')

    if split == 'train':
        n = int(len(folder_names) * pos_ratio)
    else:
        n = len(folder_names)
    for folder_name in folder_names[:n]:
        folder = base_folder + folder_name + '/'
        filenames = sorted(os.listdir(folder))
        image_paths = []
        for filename in filenames:
            if filename != '.DS_Store':
                image_paths.append(os.path.join(folder, filename))
        image_paths_list.append(image_paths)
        labels.append(1.0)
    base_folder = 'data/vilp/' + dataset + '/' + split + '/false/'
    if split == 'train':
        n = int(len(folder_names) * neg_ratio)
    else:
        n = len(folder_names)
    folder_names = sorted(os.listdir(base_folder))
    if '.DS_Store' in folder_names:
        folder_names.remove('.DS_Store')
    for folder_name in folder_names[:n]:
        folder = base_folder + folder_name + '/'
        filenames = sorted(os.listdir(folder))
        image_paths = []
        for filename in filenames:
            if filename != '.DS_Store':
                image_paths.append(os.path.join(folder, filename))
        image_paths_list.append(image_paths)
        labels.append(0.0)
    return image_paths_list, labels


def load_images_and_labels_positive(dataset='member', split='train', base=None):
    """Load image paths and labels for clevr-hans dataset.
    """
    image_paths_list = []
    labels = []
    base_folder = 'data/vilp/' + dataset + '/' + split + '/true/'
    folder_names = sorted(os.listdir(base_folder))
    if '.DS_Store' in folder_names:
        folder_names.remove('.DS_Store')

    for folder_name in folder_names:
        folder = base_folder + folder_name + '/'
        filenames = sorted(os.listdir(folder))
        image_paths = []
        for filename in filenames:
            if filename != '.DS_Store':
                image_paths.append(os.path.join(folder, filename))
                image_paths_list.append(image_paths)
                labels.append(1.0)
    return image_paths_list, labels

def load_image_clevr(path):
    """Load an image using given path.
    """
    img = cv2.imread(path)  # BGR
    assert img is not None, 'Image Not Found ' + path
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and HWC to CHW
    img = np.ascontiguousarray(img)
    return img


class VisualILP(torch.utils.data.Dataset):
    """CLEVRHans dataset. 
    The implementations is mainly from https://github.com/ml-research/NeSyConceptLearner/blob/main/src/pretrain-slot-attention/data.py.
    """

    def __init__(self, dataset, split, img_size=128, base=None, pos_ratio=1.0, neg_ratio=1.0):
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
        self.pos_ratio = pos_ratio
        self.neg_ratio = neg_ratio
        self.image_paths, self.labels = load_images_and_labels(
            dataset=dataset, split=split, base=base, pos_ratio=pos_ratio, neg_ratio=neg_ratio)

    def __getitem__(self, item):
        paths = self.image_paths[item]
        images = []
        for path in paths:
            image = Image.open(path).convert("RGB")
            image = transforms.ToTensor()(image)[:3, :, :]
            image = self.transform(image)
            image = (image - 0.5) * 2.0  # Rescale to [-1, 1].
            images.append(image)
            # TODO: concate and return??
        image = torch.stack(images, dim=0)
        return image, self.labels[item]

    def __len__(self):
        return len(self.labels)

class VisualILP_POSITIVE(torch.utils.data.Dataset):
    """CLEVRHans dataset.
    The implementations is mainly from https://github.com/ml-research/NeSyConceptLearner/blob/main/src/pretrain-slot-attention/data.py.
    """

    def __init__(self, dataset, split, img_size=128, base=None):
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
        self.image_paths, self.labels = load_images_and_labels_positive(
            dataset=dataset, split=split, base=base)

    def __getitem__(self, item):
        paths = self.image_paths[item]
        images = []
        for path in paths:
            image = Image.open(path).convert("RGB")
            image = transforms.ToTensor()(image)[:3, :, :]
            image = self.transform(image)
            image = (image - 0.5) * 2.0  # Rescale to [-1, 1].
            images.append(image)
            # TODO: concate and return??
        image = torch.stack(images, dim=0)
        return image, self.labels[item]

    def __len__(self):
        return len(self.labels)
