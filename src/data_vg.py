import json
import pickle
import random

import torch
import visual_genome.local as vg
from groundingdino.util.inference import load_image

from neumann.fol.language import DataType, Language
from neumann.fol.logic import Atom, Const, NeuralPredicate, Predicate


def to_object_type_atoms(atoms, lang):
    # rewrite the VG atoms to object-type format.
    # sitting_on(grey_cat,chair) -> sitting_on(obj1,obj2), type(obj1,grey_cat), type(obj2,chair)
    vg_constants = []
    for atom in atoms:
        # collect type constants
        # assign obj_i to each type
        # VG atoms are all types
        vg_constants.extend(atom.terms)
    vg_constants = list(set(vg_constants))

    new_lang = Language(
        preds=lang.preds.copy(), funcs=[], consts=[]
    )  # lang.consts.copy())
    new_const_dic = {}
    for i, vg_const in enumerate(vg_constants):
        object_id = vg_const.name.split("_")[-1]
        obj_const = Const("obj_{}".format(object_id), dtype=DataType("object"))
        # update language consts
        # if obj_const not in new_lang.consts:
        new_lang.consts.append(obj_const)
        new_const_dic[vg_const] = obj_const

    new_atoms = []
    p_type = NeuralPredicate("type", 2, [DataType("object"), DataType("type")])
    for atom in atoms:
        vg_terms = atom.terms
        new_obj_terms = [new_const_dic[term] for term in vg_terms]
        obj_obj_atom = Atom(atom.pred, new_obj_terms)

        obj_type_atoms = [
            Atom(p_type, [obj_const, type_const])
            for type_const, obj_const in new_const_dic.items()
        ]
        obj_type_atoms = []
        for type_const, obj_const in new_const_dic.items():
            type_const_without_id = Const(
                type_const.name.split("_")[0], dtype=DataType("type")
            )
            if not type_const_without_id in new_lang.consts:
                new_lang.consts.append(type_const_without_id)
            obj_type_atoms.append(Atom(p_type, [obj_const, type_const_without_id]))
        new_atoms.append(obj_obj_atom)
        new_atoms.extend(obj_type_atoms)
    #
    # # add target predicates and atoms
    # p_target = Predicate("target", 1, [DataType("object")])
    # for const in new_lang.consts:
    #     if const.dtype.name == "object" and "obj_" in const.name:
    #         # add e.g. target(obj_1234)
    #         target_atom = Atom(p_target, [const])
    #         if target_atom not in new_atoms:
    #             new_atoms.append(target_atom)

    new_lang = Language(
        preds=list(set(new_lang.preds)), funcs=[], consts=list(set(new_lang.consts))
    )
    return new_atoms, new_lang


class DeicticVisualGenome(torch.utils.data.Dataset):
    """Deictic Visual Genome dataset."""

    def __init__(self, path):
        self.json_data = self.load_json(path)

    def load_json(self, path):
        with open(path, "r") as f:
            json_data = json.load(f)
        return json_data

    def __getitem__(self, item):
        data = self.json_data["queries"][item]

        deictic_representation = data[0] + "."
        answer = data[1]
        image_id = data[2]
        vg_data_index = data[3]
        id = data[4]

        image_source, image = load_image(
            "data/visual_genome/VG_100K/{}.jpg".format(image_id)
        )

        # return image as one image, not two or more
        return (
            id,
            vg_data_index,
            image_id,
            image_source,
            image,
            deictic_representation,
            answer,
        )

    def __len__(self):
        return len(self.json_data["queries"])


class DeicticVisualGenomeSGGTraining(torch.utils.data.Dataset):
    """Deictic Visual Genome dataset for scene graph generator training."""

    def __init__(self, args, mode="train"):
        if mode == "train":
            self.json_data = self.load_json(
                "data/learning_deivg/deictic_vg_comp{}_sgg_train.json".format(args.complexity)
            )
        elif mode == "val":
            self.json_data = self.load_json(
                "data/learning_deivg/deictic_vg_comp{}_sgg_val.json".format(args.complexity)
            )
        elif mode == "test":
            self.json_data = self.load_json(
                "data/learning_deivg/deictic_vg_comp{}_sgg_test.json".format(args.complexity)
            )

    def load_json(self, path):
        with open(path, "r") as f:
            json_data = json.load(f)
        # randomize the order !! this causes a bug, data index should be shuffled too, that links scene graphs
        # random.shuffle(json_data['queries'])
        return json_data

    def __getitem__(self, item):
        data = self.json_data["queries"][item]
        # image_id = data["image_id"]
        # image_source, image = load_image(
        #     "data/visual_genome/VG_100K/{}.jpg".format(image_id)
        # )
        # deictic_representation = data["deictic_representation"]
        # answer = data["answer"]
        # vg_data_index = data["vg_data_index"]
        # id = data["id"]

        deictic_representation = data[0] + "."
        answer = data[1]
        image_id = data[2]
        vg_data_index = data[3]
        id = data[4]

        image_source, image = load_image(
            "data/visual_genome/VG_100K/{}.jpg".format(image_id)
        )

        # return image as one image, not two or more
        return (
            id,
            vg_data_index,
            image_id,
            image_source,
            image,
            deictic_representation,
            answer,
        )

    def __len__(self):
        return len(self.json_data["queries"])


class VisualGenomeUtils:
    """A utility class of Visual Genome dataset."""

    def __init__(self):
        self.all_relationships = self._load_relationships()
        self.all_objects = self._load_objects()

    def _load_relationships(self):
        f = open("data/visual_genome/relationships_deictic.json")
        # f = open("data/visual_genome/relationships_small.json")
        print("Loading relationships.json ...")
        rels = json.load(f)
        print("Completed.")
        f.close()
        return rels

    def _load_objects(self):
        f = open("data/visual_genome/objects.json")
        # f = open("data/visual_genome/objects_small.json")
        print("Loading objects.json ...")
        objs = json.load(f)
        print("Completed.")
        f.close()
        return objs

    def load_scene_graphs(self, num_images=20):
        scene_graphs = vg.get_scene_graphs(
            start_index=0,
            end_index=num_images,
            min_rels=1,
            data_dir="data/visual_genome/",
            image_data_dir="data/visual_genome/by-id/",
        )
        return scene_graphs

    def load_scene_graph_by_id(self, image_id):
        scene_graph = vg.get_scene_graph(
            image_id=image_id,
            images="data/visual_genome/",
            image_data_dir="data/visual_genome/by-id/",
            synset_file="data/visual_genome/synsets.json",
        )
        return scene_graph

    def graph_to_logic(self, scene_graph):
        pass

    def scene_graph_to_language(
        self, scene_graph, text, logic_generator, num_objects=3
    ):
        """Generate a FOL language out of a scene graph.

        Args:
            scene_graph (graph): A scene graph.
            text (str): A deictic prompt.
            logic_generator (_type_): _description_
            num_objects (int, optional): _description_. Defaults to 3.

        Returns:
            neumann.fol.language.Language: A FOL language.
            str: A string that represents a set of constants.
            str: A string that represents a set of predicates.
        """
        objects = list(set([str(obj).replace(" ", "") for obj in scene_graph.objects]))
        datatype = DataType("type")
        constants = [Const(obj, datatype) for obj in objects]

        const_response = "Constants:\ntype:"

        for obj in objects:
            const_response += obj
            const_response += ","
        const_response = const_response[:-1]

        predicates, pred_response = logic_generator.generate_predicates(
            text, const_response
        )
        print("Predicate generator response:\n    {}".format(pred_response))
        lang = Language(
            consts=list(set(constants)), preds=list(set(predicates)), funcs=[]
        )
        return lang, const_response, pred_response

    def data_index_to_atoms(self, data_index, lang):
        """Generate atoms given data. Relations cannot be parsed by given languageare discarded.

        Args:
            data_index (int): A data index.
            lang (neumann.fol.language.Language): A FOL language.
            rules (list[neumann.fol.logic.Clause]): A set of rules.
        Returns:
            list[neumann.fol.logic.Atom]: A list of atoms.
            neumann.fol.language.Language: An updated FOL language.
        """
        relationships = self.all_relationships[data_index]["relationships"]
        # extract predicates from rules (deictic prompt) so we keep only relavant information about the prompt. Discurd inrelavant information (atoms).
        # necessary_preds = self.extract_necessary_preds(rules)
        atoms = []
        for rel in relationships:
            atom = self.parse_relationship_by_lang(rel, lang)
            # pred can be different with respect to is_neural
            if atom != None:  # and atom.pred.name in necessary_preds:
                atoms.append(atom)
        new_atoms, new_lang = to_object_type_atoms(atoms, lang)
        return new_atoms, new_lang

        # update scene graph lang by pruning redundant temrs, i.e. keep only entities that appear in scene graph atoms
        # new_lang_pruned = self.update_lang(new_lang, new_atoms)
        # return new_atoms, new_lang_pruned

    # def extract_necessary_preds(self, rules):
    #     preds = []
    #     for rule in rules:
    #         atoms = [rule.head] + rule.body
    #         for atom in atoms:
    #             preds.append(atom.pred.name)
    #     return list(set(preds))

    # def update_lang(self, new_lang, new_atoms):
    #     all_consts = []
    #     for atom in new_atoms:
    #         # asuume only grounded atoms (facts)
    #         consts = [c for c in atom.terms]
    #         all_consts.extend(consts)
    #     all_consts = list(set(all_consts))
    #     updated_lang = Language(
    #         consts=all_consts, funcs=[], preds=new_lang.preds.copy()
    #     )
    #     return updated_lang

    def parse_relationship_by_lang(self, rel, lang):
        predicate = rel["predicate"].replace(" ", "_").lower()
        if "name" in rel["subject"].keys():
            arg_1 = rel["subject"]["name"].replace(" ", "_").lower()
        else:
            arg_1 = rel["subject"]["names"][0].replace(" ", "_").lower()

        object_id_1 = rel["subject"]["object_id"]

        if "name" in rel["object"].keys():
            arg_2 = rel["object"]["name"].replace(" ", "").lower()
        else:
            arg_2 = rel["object"]["names"][0].replace(" ", "_").lower()
        object_id_2 = rel["object"]["object_id"]

        all_preds = [p.name for p in lang.preds]
        dtype = DataType("object")

        if True:  # predicate in all_preds:
            pred = Predicate(predicate, 2, [dtype, dtype])
            const_1 = Const(
                arg_1 + "_{}".format(object_id_1),
                dtype=DataType("object_vg_{}".format(object_id_1)),
            )
            const_2 = Const(
                arg_2 + "_{}".format(object_id_2),
                dtype=DataType("object_vg_{}".format(object_id_2)),
            )
            atom = Atom(pred, [const_1, const_2])
            return atom
        else:
            return None

    def parse_relationship(self, rel):
        predicate = rel["predicate"].replace(" ", "_").lower()
        if "name" in rel["subject"].keys():
            arg_1 = rel["subject"]["name"].replace(" ", "_").lower()
        else:
            arg_1 = rel["subject"]["names"][0].replace(" ", "_").lower()

        object_id_1 = rel["subject"]["object_id"]

        if "name" in rel["object"].keys():
            arg_2 = rel["object"]["name"].replace(" ", "").lower()
        else:
            arg_2 = rel["object"]["names"][0].replace(" ", "_").lower()
        object_id_2 = rel["object"]["object_id"]

        dtype = DataType("object")

        pred = Predicate(predicate, 2, [dtype, dtype])
        const_1 = Const(
            arg_1 + "_{}".format(object_id_1),
            dtype=DataType("object_vg_{}".format(object_id_1)),
        )
        const_2 = Const(
            arg_2 + "_{}".format(object_id_2),
            dtype=DataType("object_vg_{}".format(object_id_2)),
        )
        atom = Atom(pred, [const_1, const_2])
        return atom

    def target_atoms_to_regions(self, target_atoms, data_index):
        objects_data_list = self.all_objects[data_index]["objects"]
        target_regions = []
        for target_atom in target_atoms:
            object_id = int(target_atom.terms[0].name.split("_")[-1])
            for object_data in objects_data_list:
                if object_data["object_id"] == object_id:
                    w = object_data["w"]
                    h = object_data["h"]
                    x = object_data["x"]
                    y = object_data["y"]
                    region = (x, y, w, h)
                    target_regions.append(region)
        return target_regions

    def regions_to_crop(self, image, region):
        x, y, w, h = region
        return image[x : x + w, y : y + h]


class PredictedSceneGraphUtils:
    def __init__(self, model="veto", base_path=""):
        self.model = model
        self.base_path = base_path
        self.scene_graphs = self.load_predicted_scene_graphs(model)
        self.all_relationships = self.preprocess_relationships(self.scene_graphs)

    def load_predicted_scene_graphs(self, model):
        print("Loading predicted scene graphs, model: {} ".format(model))
        with open(
            self.base_path + "data/predicted_scene_graphs/{}.pkl".format(model), "rb"
        ) as f:
            graphs = pickle.load(f)
            # rels = rels["rel_info"]["spo"]
            return graphs

    def preprocess_relationships(self, scene_graphs):
        # assing VG data index to each entry
        # build a dictionary ["data_index -> "relations"]
        all_relationsips = {}
        for graph_ in scene_graphs:
            graph = graph_[0]
            rels = graph["rel_info"]["spo"]
            img_info = graph["img_info"]
            # e.g. 'img_info': '/storage-01/ml-gsudhakaran/data/vg/VG_100K/images/2343729.jpg',
            vg_image_id = int(img_info.split("/")[-1].split(".")[0])
            all_relationsips[vg_image_id] = rels
        return all_relationsips

    def parse_relationship(self, rel):
        """
        {'img_info': '/storage-01/ml-gsudhakaran/data/vg/VG_100K/images/2343729.jpg',
        'rel_info': {'spo': ({
           's': 85,
           'p': 1,
           'o': 136,
           'phrase': 'number on tree',
           's_str': 'number',
           'o_str': 'tree',
           'p_str': 'on',
           'sbox': [403.3203125, 264.16015625, 457.51953125, 279.2969055175781],
           'obox': [461.9140625, 127.44140625, 499.0000305175781, 179.6875],
           'score': 0.9999939203262329,
           's_unique_id': 2,
           'o_unique_id': 5},
        ..."""
        dtype = DataType("object")
        pred = Predicate(rel["p_str"].replace(" ", "_").lower(), 2, [dtype, dtype])
        object_id_1 = rel["s_unique_id"]
        object_id_2 = rel["o_unique_id"]
        arg_1 = rel["s_str"].replace(" ", "_").lower()
        arg_2 = rel["o_str"].replace(" ", "_").lower()
        const_1 = Const(
            arg_1 + "_{}".format(object_id_1),
            dtype=DataType("object_vg_{}".format(object_id_1)),
        )
        const_2 = Const(
            arg_2 + "_{}".format(object_id_2),
            dtype=DataType("object_vg_{}".format(object_id_2)),
        )
        atom = Atom(pred, [const_1, const_2])
        score = rel["score"]
        return atom, score

    def load_scene_graph_by_id(self, image_id):
        scene_graph = self.all_relationships[image_id]
        return scene_graph

    def data_index_to_atoms(self, data_index, lang):
        """Generate atoms given data. Relations cannot be pfarsed by given languageare discarded.

        Args:
            data_index (int): A data index.
            lang (neumann.fol.language.Language): A FOL language.
            rules (list[neumann.fol.logic.Clause]): A set of rules.
        Returns:
            list[neumann.fol.logic.Atom]: A list of atoms.
            neumann.fol.language.Language: An updated FOL language.
        """
        rels = self.scene_graphs[data_index][0]["rel_info"]["spo"]

        atoms = []
        for rel in rels:
            atom, score = self.parse_relationship(rel)
            # pred can be different with respect to is_neural
            if atom != None and score > 0.98:  # and atom.pred.name in necessary_preds:
                atoms.append(atom)
        new_atoms, new_lang = to_object_type_atoms(atoms, lang)
        return new_atoms, new_lang

    def image_id_to_atoms(self, image_id, lang):
        """Generate atoms given data. Relations cannot be parsed by given languageare discarded.

        Args:
            data_index (int): A data index.
            lang (neumann.fol.language.Language): A FOL language.
            rules (list[neumann.fol.logic.Clause]): A set of rules.
        Returns:
            list[neumann.fol.logic.Atom]: A list of atoms.
            neumann.fol.language.Language: An updated FOL language.
        """
        # scene_graph = self.scene_graphs[data_index]
        # relationships = scene_graph[0][0]["rel_info"]["spo"]
        rels = self.all_relationships[image_id]
        # rels = self.scene_graphs[data_index][0]["rel_info"]["spo"]

        # extract predicates from rules (deictic prompt) so we keep only relavant information about the prompt. Discurd inrelavant information (atoms).
        # necessary_preds = self.extract_necessary_preds(rules)
        atoms = []
        for rel in rels:
            atom, score = self.parse_relationship(rel)
            # pred can be different with respect to is_neural
            # if atom != None and score > 0.98:  # and atom.pred.name in necessary_preds:
            if atom != None:
                atoms.append(atom)
        print(len(atoms), "atoms generated by scene graph.")
        new_atoms, new_lang = to_object_type_atoms(atoms, lang)
        return new_atoms, new_lang
