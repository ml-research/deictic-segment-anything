import json

import torch
from groundingdino.util.inference import load_image



class ArtVisualGenome(torch.utils.data.Dataset):
    """Deictic Visual Genome dataset."""

    def __init__(self, path="data/art/vg/sampled_predictions_vg.pth"):
        self.predictions = torch.load(path)
        # self.scene_graphs = self.load_json("data/gqa/val_sceneGraphs.json")
        # self.predictions = self.test_data[2]


    def load_json(self, path):
        with open(path, "r") as f:
            json_data = json.load(f)
        return json_data

    def _to_deictic_prompt(self, raw_text):
        # Input: Yes, table behind girl
        # Output: An object behind girl.
        splitted = raw_text.split(",")
        assert len(splitted) == 2, "Invalid text in instructart."
        relation_text = splitted[1]
        splitted_relation = relation_text.split(" ")
        object_words = splitted_relation[2:]
        words = ['An', 'object'] + object_words
        text =  " ".join(words)
        text = text + "."
        return text
    
    def _extract_target_name(self, raw_text):
        # Input: Yes, man wearing shorts
        # Output: man
        splitted = raw_text.split(",")
        assert len(splitted) == 2, "Invalid text in instructart."
        relation_text = splitted[1]
        splitted_relation = relation_text.split(" ")
        target_object = splitted_relation[1]
        return target_object
    
    def _extract_dummy_target_name(self, raw_text):
        # Input: Yes, man wearing shorts
        # Output: man
        return raw_text.split(" ")[-1]
        # assert len(splitted) == 2, "Invalid text in instructart."
        # relation_text = splitted[1]
        # splitted_relation = relation_text.split(" ")
        # target_object = splitted_relation[1]
        # return target_object
    
    def _get_answer_bbox(self, scene_graph, target_object):
        for obj in scene_graph["objects"].values():
            if obj["name"] == target_object \
                or obj["name"] == target_object + "s" or obj["name"] == target_object + "es" or obj["name"] == target_object + "ies"\
                    or obj["name"] + "s" == target_object or obj["name"] + "es" == target_object or obj["name"] + "ies" == target_object:
                x = obj["x"]
                y = obj["y"]
                w = obj["w"]
                h = obj["h"]
                # return torch.tensor([x, y, w, h])
                return [x, y, x + w, y + h]
                # return torch.tensor(obj["bbox"])
        return None
    
    
    def _load_art_prediction(self, index):
        """Load a single prediction from ART result."""
        item = index
        raw_text = self.predictions[0][item] # e.g. Yes, table behind girl
        deictic_representation = self._to_deictic_prompt(raw_text)
        image_id = str(self.predictions[2][item])
        subject_box = self.predictions[3][item]# / 1024
        object_box = self.predictions[4][item]# / 1024


        # scene_graph = self.scene_graphs[image_id]
        # object_box = torch.tensor(self.predictions[4][item])# / 1024
        
        # extract target bbox from scene graph
        target_label = self._extract_target_name(raw_text)
        dummy_target_label = self._extract_dummy_target_name(raw_text)
        # answer = self._get_answer_bbox(scene_graph, target_label)
        answer = subject_box
        dummy_answer = object_box
        return deictic_representation, image_id, subject_box, answer, target_label, dummy_answer
        
    def __getitem__(self, item):
        # if item < 9666:
        #     return 111, None, None, None, None, None, None
        deictic_representation, image_id, subject_box, answer, target_label, dummy_answer = self._load_art_prediction(item)
        image_source, image = load_image(f"data/visual_genome/VG_100K/{image_id}.jpg")
        image_path = f"data/visual_genome/VG_100K/{image_id}.jpg"
        return image_id, image_source, image, deictic_representation, subject_box, answer, target_label, image_path, dummy_answer

    def __len__(self):
        return len(self.predictions[0])
    
    
class ArtGQA(torch.utils.data.Dataset):
    """Deictic Visual Genome dataset."""

    def __init__(self, path="data/art/gqa/sampled_predictions_gqa.pth"):
        self.predictions = torch.load(path)
        self.scene_graphs = self.load_json("data/art/gqa/val_sceneGraphs.json")
        # self.predictions = self.test_data[2]

    def load_json(self, path):
        with open(path, "r") as f:
            json_data = json.load(f)
        return json_data
    
    def _to_deictic_prompt(self, raw_text):
        # Input: Yes, table behind chair
        # Output: An object behind chair.
        splitted = raw_text.split(",")
        assert len(splitted) == 2, "Invalid text in instructart."
        relation_text = splitted[1]
        splitted_relation = relation_text.split(" ")
        object_words = splitted_relation[2:]
        words = ['An', 'object'] + object_words
        text =  " ".join(words)
        text = text + "."
        return text
    
    # def _to_DeiGQA_instance(self, scene_graph, art_text, image_id):
    #     # scene_graph[str(image_id)]
    #     deictic_representation = self._to_deictic_prompt(art_text)
    #     answer = 
        

    def _to_deictic_prompt(self, raw_text):
        # Input: Yes, table behind girl
        # Output: An object behind girl.
        splitted = raw_text.split(",")
        assert len(splitted) == 2, "Invalid text in instructart."
        relation_text = splitted[1]
        splitted_relation = relation_text.split(" ")
        object_words = splitted_relation[2:]
        words = ['An', 'object'] + object_words
        text =  " ".join(words)
        text = text + "."
        return text
    
    def _extract_target_name(self, raw_text):
        # Input: Yes, man wearing shorts
        # Output: man
        splitted = raw_text.split(",")
        assert len(splitted) == 2, "Invalid text in instructart."
        relation_text = splitted[1]
        splitted_relation = relation_text.split(" ")
        target_object = splitted_relation[1]
        return target_object
    
    def _extract_dummy_target_name(self, raw_text):
        # Input: Yes, man wearing shorts
        # Output: man
        return raw_text.split(" ")[-1]
    
    def _get_answer_bbox(self, scene_graph, target_object):
        for obj in scene_graph["objects"].values():
            if obj["name"] == target_object \
                or obj["name"] == target_object + "s" or obj["name"] == target_object + "es" or obj["name"] == target_object + "ies"\
                    or obj["name"] + "s" == target_object or obj["name"] + "es" == target_object or obj["name"] + "ies" == target_object:
                x = obj["x"]
                y = obj["y"]
                w = obj["w"]
                h = obj["h"]
                # return torch.tensor([x, y, w, h])
                return [x, y, x + w, y + h]
                # return torch.tensor(obj["bbox"])
        return None
    
    # def _get_answer_bbox(self, scene_graph, target_object):
    #     for obj in scene_graph["objects"].values():
    #         if obj["name"] == target_object \
    #             or obj["name"] == target_object + "s" or obj["name"] == target_object + "es" or obj["name"] == target_object + "ies"\
    #                 or obj["name"] + "s" == target_object or obj["name"] + "es" == target_object or obj["name"] + "ies" == target_object:
    #             x = obj["x"]
    #             y = obj["y"]
    #             w = obj["w"]
    #             h = obj["h"]
    #             # return torch.tensor([x, y, w, h])
    #             return [x, y, x + w, y + h]
    #             # return torch.tensor(obj["bbox"])
    #     return None
    
    def _load_art_prediction(self, index):
        """Load a single prediction from ART result."""
        item = index
        raw_text = self.predictions[0][item] # e.g. Yes, table behind girl
        deictic_representation = self._to_deictic_prompt(raw_text)
        image_id = str(self.predictions[2][item])
        subject_box = torch.tensor(self.predictions[3][item])# / 1024
        object_box = torch.tensor(self.predictions[4][item])# / 1024


        scene_graph = self.scene_graphs[image_id]
        # object_box = torch.tensor(self.predictions[4][item])# / 1024
        
        # extract target bbox from scene graph
        target_label = self._extract_target_name(raw_text)
        dummy_target_label = self._extract_dummy_target_name(raw_text)
        answer = self._get_answer_bbox(scene_graph, target_label)
        dummy_answer = self._get_answer_bbox(scene_graph, dummy_target_label)
        return deictic_representation, image_id, subject_box, answer, target_label, dummy_answer
        
    def __getitem__(self, item):
        # if item < 9666:
        #     return 111, None, None, None, None, None, None
        deictic_representation, image_id, subject_box, answer, target_label, dummy_answer = self._load_art_prediction(item)
        image_source, image = load_image(
            "data/art/gqa/images/{}.jpg".format(image_id)
        )
        image_path = "data/art/gqa/images/{}.jpg".format(image_id)
        return image_id, image_source, image, deictic_representation, subject_box, answer, target_label, image_path, dummy_answer
    def __len__(self):
        return len(self.predictions[0])
    
