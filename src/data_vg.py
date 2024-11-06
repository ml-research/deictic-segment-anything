import json
import pickle
import torch
import random

import visual_genome.local as vg
from groundingdino.util.inference import load_image
from neumann.fol.language import DataType, Language
from neumann.fol.logic import Atom, Const, NeuralPredicate, Predicate


def to_object_type_atoms(atoms, lang):
    """Convert VG atoms to an object-type format."""
    unique_constants = set(term for atom in atoms for term in atom.terms)
    new_lang = Language(preds=lang.preds.copy(), funcs=[], consts=[])
    new_const_dic = {
        vg_const: Const(f"obj_{vg_const.name.split('_')[-1]}", dtype=DataType("object"))
        for vg_const in unique_constants
    }

    new_lang.consts.extend(new_const_dic.values())
    p_type = NeuralPredicate("type", 2, [DataType("object"), DataType("type")])
    
    new_atoms = [
        Atom(atom.pred, [new_const_dic[term] for term in atom.terms])
        for atom in atoms
    ]

    for type_const, obj_const in new_const_dic.items():
        base_type_const = Const(type_const.name.split("_")[0], dtype=DataType("type"))
        if base_type_const not in new_lang.consts:
            new_lang.consts.append(base_type_const)
        new_atoms.append(Atom(p_type, [obj_const, base_type_const]))

    new_lang = Language(
        preds=list(set(new_lang.preds)), funcs=[], consts=list(set(new_lang.consts))
    )
    return new_atoms, new_lang


class BaseVisualGenomeDataset(torch.utils.data.Dataset):
    """Base class for Visual Genome datasets."""

    def __init__(self, json_path):
        self.json_data = self._load_json(json_path)

    def _load_json(self, path):
        with open(path, "r") as file:
            return json.load(file)

    def _parse_data(self, item):
        data = self.json_data["queries"][item]
        return data[0] + ".", data[1], data[2], data[3], data[4]

    def __len__(self):
        return len(self.json_data["queries"])


class DeicticVisualGenome(BaseVisualGenomeDataset):
    """Deictic Visual Genome dataset."""

    def __getitem__(self, item):
        deictic_representation, answer, image_id, vg_data_index, id = self._parse_data(item)
        image_source, image = load_image(f"data/visual_genome/VG_100K/{image_id}.jpg")
        return id, vg_data_index, image_id, image_source, image, deictic_representation, answer


class DeicticVisualGenomeSGGTraining(BaseVisualGenomeDataset):
    """Deictic Visual Genome dataset for scene graph generator training."""

    def __init__(self, args, mode="train"):
        filename = f"deictic_vg_comp{args.complexity}_sgg_{mode}.json"
        json_path = f"data/deivg_learning/{filename}"
        super().__init__(json_path)

    def __getitem__(self, item):
        deictic_representation, answer, image_id, vg_data_index, id = self._parse_data(item)
        image_source, image = load_image(f"data/visual_genome/VG_100K/{image_id}.jpg")
        return id, vg_data_index, image_id, image_source, image, deictic_representation, answer


class VisualGenomeUtils:
    """A utility class for the Visual Genome dataset."""

    def __init__(self):
        self.all_relationships = self._load_json("data/visual_genome/relationships_deictic.json")
        self.all_objects = self._load_json("data/visual_genome/objects.json")

    def _load_json(self, path):
        print(f"Loading {path} ...")
        with open(path, "r") as file:
            data = json.load(file)
        print("Completed.")
        return data

    def load_scene_graphs(self, num_images=20):
        return vg.get_scene_graphs(
            start_index=0,
            end_index=num_images,
            min_rels=1,
            data_dir="data/visual_genome/",
            image_data_dir="data/visual_genome/by-id/",
        )

    def load_scene_graph_by_id(self, image_id):
        return vg.get_scene_graph(
            image_id=image_id,
            images="data/visual_genome/",
            image_data_dir="data/visual_genome/by-id/",
            synset_file="data/visual_genome/synsets.json",
        )

    def scene_graph_to_language(self, scene_graph, text, logic_generator, num_objects=3):
        """Generate FOL language from a scene graph."""
        objects = list(set(obj.replace(" ", "") for obj in scene_graph.objects))
        datatype = DataType("type")
        constants = [Const(obj, datatype) for obj in objects]

        const_response = f"Constants:\ntype:{','.join(objects)}"
        predicates, pred_response = logic_generator.generate_predicates(text, const_response)
        lang = Language(consts=list(set(constants)), preds=list(set(predicates)), funcs=[])
        return lang, const_response, pred_response

    def data_index_to_atoms(self, data_index, lang):
        """Generate atoms from data index."""
        relationships = self.all_relationships[data_index]["relationships"]
        atoms = [self._parse_relationship(rel, lang) for rel in relationships if self._parse_relationship(rel, lang)]
        return to_object_type_atoms(atoms, lang)

    def _parse_relationship(self, rel, lang):
        pred_name = rel["predicate"].replace(" ", "_").lower()
        dtype = DataType("object")
        pred = Predicate(pred_name, 2, [dtype, dtype])
        # Either of name of names is used as a key
        # Add key "name" if it is names
        if "names" in rel["object"].keys():
            rel["object"]["name"] = rel["object"]["names"][0]
        if "names" in rel["subject"].keys():
            rel["subject"]["name"] = rel["subject"]["names"][0]
        consts = [
            Const(rel["subject"]["name"].replace(" ", "_").lower() + f"_{rel['subject']['object_id']}", dtype=dtype),
            Const(rel["object"]["name"].replace(" ", "").lower() + f"_{rel['object']['object_id']}", dtype=dtype),
        ]
        return Atom(pred, consts)


class PredictedSceneGraphUtils:
    """Utils for predicted scene graphs."""

    def __init__(self, model="veto", base_path=""):
        self.model = model
        self.base_path = base_path
        self.scene_graphs = self._load_predicted_scene_graphs(model)
        self.all_relationships = self._preprocess_relationships(self.scene_graphs)

    def _load_predicted_scene_graphs(self, model):
        with open(f"{self.base_path}data/predicted_scene_graphs/{model}.pkl", "rb") as file:
            return pickle.load(file)

    def _preprocess_relationships(self, scene_graphs):
        return {
            int(graph_[0]["img_info"].split("/")[-1].split(".")[0]): graph_[0]["rel_info"]["spo"]
            for graph_ in scene_graphs
        }

    def _parse_relationship(self, rel):
        dtype = DataType("object")
        pred = Predicate(rel["p_str"].replace(" ", "_").lower(), 2, [dtype, dtype])
        consts = [
            Const(rel["s_str"].replace(" ", "_").lower() + f"_{rel['s_unique_id']}", dtype=dtype),
            Const(rel["o_str"].replace(" ", "_").lower() + f"_{rel['o_unique_id']}", dtype=dtype),
        ]
        return Atom(pred, consts), rel["score"]

    def load_scene_graph_by_id(self, image_id):
        return self.all_relationships.get(image_id, [])

    def data_index_to_atoms(self, data_index, lang):
        return self._generate_atoms(lang, self.scene_graphs[data_index][0]["rel_info"]["spo"])

    def image_id_to_atoms(self, image_id, lang):
        return self._generate_atoms(lang, self.all_relationships.get(image_id, []))

    def _generate_atoms(self, lang, relationships):
        atoms = [
            self._parse_relationship(rel)[0]
            for rel in relationships
            if self._parse_relationship(rel)[1] > 0.98
        ]
        return to_object_type_atoms(atoms, lang)