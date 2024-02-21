import json
import os
import random

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image


def load_image_clevr(path):
    """Load an image using given path.
    """
    img = cv2.imread(path)  # BGR
    assert img is not None, 'Image Not Found ' + path
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and HWC to CHW
    img = np.ascontiguousarray(img)
    return img

def load_question_json(question_json_path):
    with open(question_json_path) as f:
        question = json.load(f)
    #questions = question["questions"]
    #return questions
    return question

class BehindTheScenes(torch.utils.data.Dataset):
    def __init__(self, question_json_path, lang, n_data, device,  img_size=128, base='data/behind-the-scenes/'):
        super().__init__()
        #self.colors = ["cyan", "blue", "yellow",\
        #               "purple", "red", "green", "gray", "brown"]
        self.colors = ["cyan", "gray", "red", "yellow"]
        self.query_types = ["delete", "append", "reverse", "sort"]
        self.positions = ["1st", "2nd", "3rd"]
        self.base = base
        self.lang = lang
        self.device = device
        self.questions = random.sample(load_question_json(question_json_path), int(n_data))
        self.img_size = img_size
        self.transform = transforms.Compose(
            [transforms.Resize((img_size, img_size))]
        )
        #self.image_paths, self.answer_paths = load_images_and_labels(
        #    dataset=dataset, split=split, base=base)
        # {"program": "query2(sort,2nd)", "split": "train", "image_index": 45, "answer": "red", \
        # "image": "BehindTheScenes_train_000045", "question_index": 10290, "question": ["sort", "2nd"], \
        # "image_filename": "BehindTheScenes_train_000045.png"},

    def __getitem__(self, item):
        question = self.questions[item]
        # print('question: ', question)
        image_path = self.base + 'images/' + question["image_filename"]
        image = Image.open(image_path).convert("RGB")
        image = self.image_preprocess(image)
        answer = self.to_onehot(self.colors.index(question["answer"]), len(self.colors))
        query_tuple = question["question"]
        query = self.to_query_vector(query_tuple)
        # TODO: concate and return??
        # answer = load_answer(self.answer_paths[item])
        return image, query, answer

    def __len__(self):
        return len(self.questions)
        
    def to_onehot(self, index, size):
        onehot = torch.zeros(size, ).to(self.device)
        onehot[index] = 1.0
        return onehot

    def to_query_vector(self, query_tuple):
        if len(query_tuple) == 3:
            query_type, color, position = query_tuple
            # ("delete", "red", "1st")
            q_1 = self.to_onehot(self.query_types.index(query_type), len(self.query_types))
            q_2 = self.to_onehot(self.colors.index(color), len(self.colors))
            q_3 = self.to_onehot(self.positions.index(position), len(self.positions))
            return torch.cat([q_1, q_2, q_3])
        elif len(query_tuple) == 2:
            query_type, position = query_tuple
            # ("sort", "1st")
            q_1 = self.to_onehot(self.query_types.index(query_type), len(self.query_types))
            q_2 = torch.zeros(len(self.colors, )).to(self.device)
            q_3 = self.to_onehot(self.positions.index(position), len(self.positions))
            return torch.cat([q_1, q_2, q_3])

    
    def image_preprocess(self, image):
        image = transforms.ToTensor()(image)[:3, :, :]
        image = self.transform(image)
        image = (image - 0.5) * 2.0  # Rescale to [-1, 1].
        return image.unsqueeze(0)

