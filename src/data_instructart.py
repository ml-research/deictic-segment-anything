import json

import torch
from groundingdino.util.inference import load_image



# class InstructArtVisualGenome(torch.utils.data.Dataset):
#     """Deictic Visual Genome dataset."""

#     def __init__(self, path="data/test_result_for_deisam.pth"):
#         self.test_data = torch.load(path)
#         self.predictions = self.test_data[2]

#     def load_json(self, path):
#         with open(path, "r") as f:
#             json_data = json.load(f)
#         return json_data

#     def _to_deictic_prompt(self, raw_text):
#         # Input: Yes, table behind girl
#         # Output: An object behind girl.
#         splitted = raw_text.split(",")
#         assert len(splitted) == 2, "Invalid text in instructart."
#         relation_text = splitted[1]
#         splitted_relation = relation_text.split(" ")
#         object_words = splitted_relation[2:]
#         words = ['An', 'object'] + object_words
#         text =  " ".join(words)
#         text = text + "."
#         return text
        
#     def __getitem__(self, item):
        
#         # for i in range(5):
#         # predictions = self.predictions[item]
#         raw_text = self.predictions[0][item] # e.g. Yes, table behind girl
#         deictic_representation = self._to_deictic_prompt(raw_text)
#         image_id = self.predictions[2][item]
#         answer = self.predictions[3][item]


#         image_source, image = load_image(
#             "data/visual_genome/VG_100K/{}.jpg".format(image_id)
#         )
        
#         vg_data_index = item

#         # return image as one image, not two or more
#         return (
#             vg_data_index,
#             image_id,
#             image_source,
#             deictic_representation,
#             answer,
#         )

#     def __len__(self):
#         return len(self.predictions[0])
    
    
class ArtGQA(torch.utils.data.Dataset):
    """Deictic Visual Genome dataset."""

    def __init__(self, path="data/gqa/test_result_gqa.pth"):
        self.test_data = torch.load(path)
        self.scene_graphs = self.load_json("data/gqa/val_sceneGraphs.json")
        self.predictions = self.test_data[2]

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
        subject_box = torch.tensor(self.predictions[3][item])# / 1024


        scene_graph = self.scene_graphs[image_id]
        # object_box = torch.tensor(self.predictions[4][item])# / 1024
        
        # extract target bbox from scene graph
        target_label = self._extract_target_name(raw_text)
        answer = self._get_answer_bbox(scene_graph, target_label)
        return deictic_representation, image_id, subject_box, answer, target_label
        
    def __getitem__(self, item):
        # if item < 9666:
        #     return 111, None, None, None, None, None, None
        deictic_representation, image_id, subject_box, answer, target_label = self._load_art_prediction(item)
        image_source, image = load_image(
            "data/gqa/images/{}.jpg".format(image_id)
        )
        return image_id, image_source, image, deictic_representation, subject_box, answer, target_label

        
    # def __getitem__(self, item):
    #     # for i in range(5):
    #     # predictions = self.predictions[item]
    #     raw_text = self.predictions[0][item] # e.g. Yes, table behind girl
    #     deictic_representation = self._to_deictic_prompt(raw_text)
    #     image_id = self.predictions[2][item]
    #     subject_box = torch.tensor(self.predictions[3][item])# / 1024
    #     scene_graph = self.scene_graphs[image_id]
    #     object_box = torch.tensor(self.predictions[4][item])# / 1024

    #     image_source, image = load_image(
    #         "data/gqa/images/{}.jpg".format(image_id)
    #     )
        
    #     vg_data_index = item

    #     # return image as one image, not two or more
    #     return (
    #         vg_data_index,
    #         image_id,
    #         image_source,
    #         deictic_representation,
    #         subject_box
    #     )

    def __len__(self):
        return len(self.predictions[0])
    
    
# class InstructArtAnnotatedGQA(torch.utils.data.Dataset):
#     """Deictic Visual Genome dataset."""

#     def __init__(self, path="data/test_result_gqa.pth"):
#         self.test_data = torch.load(path)
#         self.predictions = self.test_data[2]

#     def load_json(self, path):
#         with open(path, "r") as f:
#             json_data = json.load(f)
#         return json_data
    
#     def _to_subject(self, raw_text):
#         # Input: Yes, table behind girl
#         # Output: An object behind girl.
#         splitted = raw_text.split(",")
#         assert len(splitted) == 2, "Invalid text in instructart."
#         relation_text = splitted[1]
#         splitted_relation = relation_text.split(" ")
#         subject_word = splitted_relation[1]
#         return subject_word
        
#     def __getitem__(self, item):
        
#         # for i in range(5):
#         # predictions = self.predictions[item]
#         raw_text = self.predictions[0][item] # e.g. Yes, table behind girl
#         deictic_representation = self._to_subject(raw_text)
#         image_id = self.predictions[2][item]
#         answer = self.predictions[3][item]


#         image_source, image = load_image(
#             "data/visual_genome/VG_100K/{}.jpg".format(image_id)
#         )
        
#         vg_data_index = item

#         # return image as one image, not two or more
#         return (
#             vg_data_index,
#             image_id,
#             image_source,
#             deictic_representation,
#             answer,
#         )

#     def __len__(self):
#         return len(self.predictions[0])




# class InstructArtAnnotatedVisualGenome(torch.utils.data.Dataset):
#     """Deictic Visual Genome dataset."""

#     def __init__(self, path="data/test_result_for_deisam.pth"):
#         self.test_data = torch.load(path)
#         self.predictions = self.test_data[2]

#     def load_json(self, path):
#         with open(path, "r") as f:
#             json_data = json.load(f)
#         return json_data
    
#     def _to_subject(self, raw_text):
#         # Input: Yes, table behind girl
#         # Output: An object behind girl.
#         splitted = raw_text.split(",")
#         assert len(splitted) == 2, "Invalid text in instructart."
#         relation_text = splitted[1]
#         splitted_relation = relation_text.split(" ")
#         subject_word = splitted_relation[1]
#         return subject_word
        
#     def __getitem__(self, item):
        
#         # for i in range(5):
#         # predictions = self.predictions[item]
#         raw_text = self.predictions[0][item] # e.g. Yes, table behind girl
#         deictic_representation = self._to_subject(raw_text)
#         image_id = self.predictions[2][item]
#         answer = self.predictions[3][item]


#         image_source, image = load_image(
#             "data/visual_genome/VG_100K/{}.jpg".format(image_id)
#         )
        
#         vg_data_index = item

#         # return image as one image, not two or more
#         return (
#             vg_data_index,
#             image_id,
#             image_source,
#             deictic_representation,
#             answer,
#         )

#     def __len__(self):
#         return len(self.predictions[0])