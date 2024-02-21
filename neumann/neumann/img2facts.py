import torch
import torch.nn as nn


class Img2Facts(nn.Module):
    """Img2Facts module converts raw images into a form of probabilistic facts. Each image is fed to the perception module, and the result is concatenated and fed to facts converter.
    """

    def __init__(self, perception_module, facts_converter, atoms, bk, device):
        super().__init__()
        self.pm = perception_module
        self.fc = facts_converter
        self.atoms = atoms
        self.bk = bk
        self.device = device

    def forward(self, x):
        # x: batch_size * num_iamges * C * W * H
        num_images = x.size(1)
        # feed each input image
        zs_list = []
        # TODO: concat image ids
        for i in range(num_images):
            # B * E * D
            zs = self.pm(x[:, i, :, :])
            # image_ids = torch.tensor([i for i in range(x.size(0))]).unsqueeze(0).unsqueeze(0).to(self.device)
            image_ids = torch.tensor(i).expand(
                (zs.size(0), zs.size(1), 1)).to(self.device)
            zs = torch.cat((zs, image_ids), dim=2)
            zs_list.append(zs)
        zs = torch.cat(zs_list, dim=1)
        # zs: batch_size * num_images * num_objects * num_attributes
        return self.fc(zs)

class Img2FactsWithQuery(nn.Module):
    """Img2Facts module converts raw images into a form of probabilistic facts. Each image is fed to the perception module, and the result is concatenated and fed to facts converter.
    """

    def __init__(self, perception_module, facts_converter, atoms, bk, device):
        super().__init__()
        self.pm = perception_module
        self.fc = facts_converter
        self.atoms = atoms
        self.bk = bk
        self.device = device

    def forward(self, x, query):
        # x: batch_size * num_iamges * C * W * H
        num_images = x.size(1)
        # feed each input image
        zs_list = []
        # TODO: concat image ids
        for i in range(num_images):
            # B * E * D
            zs = self.pm(x[:, i, :, :])
            # image_ids = torch.tensor([i for i in range(x.size(0))]).unsqueeze(0).unsqueeze(0).to(self.device)
            image_ids = torch.tensor(i).expand(
                (zs.size(0), zs.size(1), 1)).to(self.device)
            zs = torch.cat((zs, image_ids), dim=2)
            zs_list.append(zs)
        zs = torch.cat(zs_list, dim=1)
        # zs: batch_size * num_images * num_objects * num_attributes
        return self.fc(zs, query)