import time
from cgitb import text
from hmac import new

import openai
import torch

# from diffusers.scripts.convert_kakao_brain_unclip_to_diffusers import text_encoder

from neumann.fol.language import Language


class SemanticUnifier:
    def __init__(self, graph_lang, device):
        # self.lang = lang
        self.graph_lang = graph_lang
        self.device = device
        self.const_mapping = {}
        self.pred_mapping = {}

        # set up embeddings
        # self.consts_embeddings = self._get_consts_embeddings()
        # self.preds_embeddings = self._get_preds_embeddings()
        # self.graph_consts_embeddings = self._get_graph_consts_embeddings()
        # self.graph_preds_embeddings = self._get_graph_preds_embeddings()

    def _init_scene_graph_consts_embeddings(self):
        self.graph_consts_embeddings = self._get_graph_consts_embeddings()

    def _init_scene_graph_preds_embeddings(self):
        self.graph_preds_embeddings = self._get_graph_preds_embeddings()

    def _get_consts_embeddings(self):
        dic = {}
        for c in self.lang.consts:
            c_embedding = self.get_embedding(c.name.replace("_", " "))
            dic[c] = c_embedding
        return dic

    def _get_preds_embeddings(self):
        dic = {}
        for p in self.lang.preds:
            p_embedding = self.get_embedding(p.name.replace("_", " "))
            dic[p] = p_embedding
        return dic

    def _get_graph_consts_embeddings(self):
        dic = {}
        for c in self.graph_lang.consts:
            c_embedding = self.get_embedding(c.name)
            dic[c] = c_embedding
        return dic

    def _get_graph_preds_embeddings(self):
        dic = {}
        for p in self.graph_lang.preds:
            p_embedding = self.get_embedding(p.name)
            dic[p] = p_embedding
        return dic

    def to_language(self, graph_atoms):
        """Generate a FOL language given atoms that represent a scene graph.

        Args:
            graph_atoms (Atom): a set of atoms represent a scene graph.

        Returns:
            Language : a language computed from graph atoms.
        """
        preds = set()
        consts = set()

        for atom in graph_atoms:
            preds.add(atom.pred)
            for c in atom.terms:
                consts.add(c)
        lang = Language(preds=list(preds), funcs=[], consts=list(consts))
        return lang

    def get_most_similar_index(self, x, ys):
        pass

    def get_most_similar_predicate_in_graph(self, pred):
        # num_graph_pred = len(self.graph_lang.preds)
        X = self.get_embedding(pred.name).unsqueeze(0)  # .expand((num_graph_pred, -1))
        X_graph = torch.stack(list(self.graph_preds_embeddings.values()))
        # score = torch.dot(X.T, X_graph)
        score = torch.sum(X * X_graph, axis=-1)
        index = torch.argmax(score).item()
        return self.graph_lang.preds[index]

    def get_most_similar_constant_in_graph(self, const):
        X = self.get_embedding(const.name).unsqueeze(0)
        X_graph = torch.stack(list(self.graph_consts_embeddings.values()))
        score = torch.sum(X * X_graph, axis=-1)
        index = torch.argmax(score).item()
        # print(self.graph_lang.consts, len(self.graph_lang.consts))
        # print(score, score.shape)
        # print(index)
        return self.graph_lang.consts[index]

    def build_const_mapping(self, lang, graph_lang):
        dic = {}
        for c in lang.consts:
            if not c in graph_lang.consts:
                # find the most similar graph const
                return 0
        pass

    def build_pred_mapping(self, lang, graph_lang):
        pass

    def rewrite_lang(self, lang, graph_lang):
        """Rewrite the language using only existing vocabluary in the graph.

        Args:
            lang (_type_): _description_

        Returns:
            _type_: _description_
        """
        new_lang = Language(
            preds=lang.preds.copy(), funcs=[], consts=lang.consts.copy()
        )
        const_mapping = {}
        for const in new_lang.consts:
            if const not in graph_lang.consts:
                new_const = self.get_most_similar_constant_in_graph(const)
                new_lang.consts.remove(const)
                new_lang.consts.append(new_const)
                const_mapping[const] = new_const
        predicate_mapping = {}
        for pred in new_lang.preds:
            if pred not in graph_lang.preds:
                new_pred = self.get_most_similar_predicate_in_graph(pred)
                new_lang.preds.remove(pred)
                new_lang.preds.append(new_pred)
                predicate_mapping[pred] = new_pred
        return new_lang, const_mapping, predicate_mapping

    # def rewrite_rules(self, rules, const_mapping, predicate_mapping):
    #     new_rules = []
    #     for rule in rules:
    #         new_rule = rule
    #         new_atoms = []
    #         for atom in [rule.head] + rule.body:
    #             # check / rewrite predicate
    #             if atom.pred in predicate_mapping.keys():
    #                 atom.pred = predicate_mapping[atom.pred]
    #             # check / rewrite const
    #             for i, const in enumerate(atom.terms):
    #                 if const in const_mapping.keys():
    #                     atom.terms[i] = const_mapping[const]
    #             new_atoms.append(atom)
    #         new_rule.head = new_atoms[0]
    #         new_rule.body = new_atoms[1:]
    #         new_rules.append(new_rule)
    #     return new_rules

    def rewrite_rules(self, rules, lang, graph_lang, rewrite_pred=True):
        reserved_preds = ["target", "type", "cond1", "cond2", "cond3"]
        self._init_scene_graph_consts_embeddings()
        self._init_scene_graph_preds_embeddings()
        new_rules = []
        # new_lang = Language(preds=lang.preds.copy(), funcs=[], consts=lang.consts.copy())
        for rule in rules:
            new_rule = rule
            new_atoms = []
            for atom in [rule.head] + rule.body:
                # check / rewrite predicate
                pred = atom.pred
                if (
                    rewrite_pred
                    and pred.name not in reserved_preds
                    and pred not in graph_lang.preds
                ):
                    # replace the non-existing predicate by the most similar one
                    new_pred = self.get_most_similar_predicate_in_graph(pred)
                    atom.pred = new_pred
                    self.pred_mapping[pred.name] = new_pred.name
                    print(pred.name, " -> ", new_pred.name)
                    # new_lang.preds.remove(pred)
                    # new_lang.preds.append(new_pred)
                # check / rewrite const
                for i, const in enumerate(atom.terms):
                    if (
                        const.__class__.__name__ == "Const"
                        and const not in graph_lang.consts
                    ):
                        # replace the non-existing constant by the most similar one
                        new_const = self.get_most_similar_constant_in_graph(const)
                        atom.terms[i] = new_const
                        self.const_mapping[const.name] = new_const.name
                        print(const.name, " -> ", new_const.name)
                        # new_lang.consts.remove(const)
                        # new_lang.consts.append(new_const)
                new_atoms.append(atom)
            new_rule.head = new_atoms[0]
            new_rule.body = new_atoms[1:]
            new_rules.append(new_rule)
        return new_rules  # , #new_lang

    # def unify(self, lang, graph_lang, rules):
    # rewrite lang
    # internally overwrite self.lang
    # new_lang, const_mapping, pred_mapping = self.rewrite_lang(lang, graph_lang)
    # generate new rules using the refined language
    # new_rules new_lang = self.rewrite_rules(rules, lang, graph_lang)
    # eturn new_lang, new_rules

    def get_embedding(self, text_to_embed):
        response = openai.Embedding.create(
            model="text-embedding-ada-002", input=[text_to_embed.replace("_", " ")]
        )
        # Extract the AI output embedding as a list of floats
        embedding = torch.tensor(response["data"][0]["embedding"]).to(self.device)
        return embedding

        # try:
        #     # Embed a line of text
        #     response = openai.Embedding.create(
        #         model="text-embedding-ada-002", input=[text_to_embed.replace("_", " ")]
        #     )
        #     # Extract the AI output embedding as a list of floats
        #     embedding = torch.tensor(response["data"][0]["embedding"]).to(self.device)
        #     return embedding
        # # except (openai.ServiceUnavailableError, openai.InvalidRequestError):
        # except (openai.InvalidRequestError, openai.error.ServiceUnavailableError):
        #     print(
        #         "Got openai.InvalidRequestError or openai.error.ServiceUnavailableError for embeddings in Semantic Unification, waiting for 3s and try again..."
        #     )
        #     time.sleep(3)
        #     return self.get_embedding(text_to_embed=text_to_embed)
