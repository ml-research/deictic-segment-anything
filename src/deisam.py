import random
import numpy as np
import torch
import lark
from torch.nn import Module
from deisam_utils import load_neumann, load_neumann_for_sgg_training, load_sam_model
from learning_utils import (
    get_target_selection_rules,
    merge_atoms_list,
    merge_langs,
    translate_atoms_to_sgg_format,
    translate_rules_to_sgg_format,
)
from llm_logic_generator import LLMLogicGenerator
from sam_utils import to_boxes, to_boxes_with_sgg, to_transformed_boxes
from semantic_unifier import SemanticUnifier
from visual_genome_utils import (
    get_init_language_with_sgg,
    scene_graph_to_language,
    scene_graph_to_language_with_sgg,
)
from neumann.torch_utils import softor


class BaseDeiSAM(Module):
    def __init__(self, api_key, device, vg_utils, sgg_model=None):
        super(BaseDeiSAM, self).__init__()
        self.device = device
        self.sam_predictor = load_sam_model(device)
        self.llm_logic_generator = LLMLogicGenerator(api_key)
        self.visual_genome_utils = vg_utils
        self.sgg_model = sgg_model

    def _generate_rules_by_llm(self, deictic_text, lang):
        try:
            return self.llm_logic_generator.generate_rules(deictic_text, lang)
        except (lark.exceptions.VisitError, lark.exceptions.UnexpectedCharacters,
                lark.exceptions.UnexpectedEOF, IndexError, RuntimeError):
            print("Failed to parse the LLM response.")
            return []

    def _unify_semantics(self, text_lang, scene_graph_lang, llm_rules):
        try:
            semantic_unifier = SemanticUnifier(scene_graph_lang, self.device)
            return semantic_unifier.rewrite_rules(
                rules=llm_rules, lang=text_lang, graph_lang=scene_graph_lang)
        except (RuntimeError, IndexError):
            return llm_rules

    def _forward_reasoning(self, lang, atoms, rules):
        try:
            print("Building NEUMANN reasoner...")
            fc, neumann = load_neumann(lang, rules, atoms, self.device)
            self.neumann = neumann
            v_0 = fc(atoms)
            v_T = neumann(v_0)
            target_atoms = [atom for atom in neumann.get_top_atoms(v_T[0]) if atom.pred.name == "target"]
            self.target_atoms = target_atoms
            return target_atoms, v_T, neumann
        except (RuntimeError, IndexError):
            self.target_atoms = []
            return [], None, None

    def _segment_objects_by_sam(self, image_source, target_atoms, transformer_func, identifier):
        boxes = transformer_func(target_atoms, identifier, self.visual_genome_utils)
        self.sam_predictor.set_image(image_source)
        transformed_boxes = to_transformed_boxes(
            boxes, image_source, self.sam_predictor, self.device)
        try:
            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords=None, point_labels=None, boxes=transformed_boxes, multimask_output=False)
            return masks
        except (RuntimeError, AttributeError):
            return []

class DeiSAM(BaseDeiSAM):
    def forward(self, data_index, image_id, scene_graph, text, image_source):
        lang = scene_graph_to_language(
            scene_graph=scene_graph, text=text, logic_generator=self.llm_logic_generator, num_objects=2)
        
        llm_rules = self._generate_rules_by_llm(text, lang)
        rewritten_rules = None

        print("LLM-generated rules:")
        for rule in llm_rules:
            print("   ", rule)

        scene_graph_atoms, scene_graph_lang = self.visual_genome_utils.data_index_to_atoms(
            data_index=data_index, lang=lang)
        
        target_atoms, v_T, neumann = self._forward_reasoning(
            scene_graph_lang, scene_graph_atoms, llm_rules)

        if not target_atoms:
            rewritten_rules = self._unify_semantics(lang, scene_graph_lang, llm_rules)
            print("Semantically unified rules:")
            for rule in rewritten_rules:
                print("   ", rule)
            target_atoms, v_T, neumann = self._forward_reasoning(
                scene_graph_lang, scene_graph_atoms, rewritten_rules)

        masks = self._segment_objects_by_sam(image_source, target_atoms, to_boxes, data_index)
        return masks, llm_rules, rewritten_rules


class DeiSAMSGG(BaseDeiSAM):
    def forward(self, data_index, image_id, scene_graph, text, image_source):
        lang = get_init_language_with_sgg(
            scene_graph=scene_graph, text=text, logic_generator=self.llm_logic_generator)

        llm_rules = self._generate_rules_by_llm(text, lang)
        rewritten_rules = None

        print("LLM-generated rules:")
        for rule in llm_rules:
            print("   ", rule)

        scene_graph_atoms, scene_graph_lang = self.visual_genome_utils.image_id_to_atoms(
            image_id=image_id, lang=lang)

        scene_graph_base_lang = scene_graph_to_language_with_sgg(scene_graph)

        target_atoms, v_T, neumann = self._forward_reasoning(
            scene_graph_lang, scene_graph_atoms, llm_rules)

        if not target_atoms:
            rewritten_rules = self._unify_semantics(lang, scene_graph_base_lang, llm_rules)
            print("Semantically unified rules:")
            for rule in rewritten_rules:
                print("   ", rule)
            target_atoms, v_T, neumann = self._forward_reasoning(
                scene_graph_lang, scene_graph_atoms, rewritten_rules)

        masks = self._segment_objects_by_sam(image_source, target_atoms, to_boxes_with_sgg, image_id)
        return masks, llm_rules, rewritten_rules


class TrainableDeiSAM(BaseDeiSAM):
    """A class of trainable DeiSAM that parameterizes scene graph generators as weighted mixtures."""

    def __init__(self, api_key, device, vg_utils_list, sem_uni=False):
        super(TrainableDeiSAM, self).__init__(api_key, device, vg_utils_list[0])
        self.visual_genome_utils_list = vg_utils_list
        self.rule_weights = self._init_random_weights(1, len(vg_utils_list), device)
        self.sem_uni = sem_uni

    def _init_random_weights(self, program_size, num_rules, device):
        """Initialize the clause weights with random values."""
        return torch.nn.Parameter(torch.Tensor(np.random.rand(program_size, num_rules)).to(device))

    def _softmax_clause_weights(self, clause_weights, temp=1e-1):
        """Apply softmax to the clause weights to choose M clauses."""
        softmaxed_weights = torch.softmax(clause_weights / temp, dim=1)
        return softor(softmaxed_weights, dim=0)

    def forward_reasoning(self, lang, atoms, rules_to_learn, rules_bk):
        """Perform forward reasoning with a GNN-based differentiable forward reasoner called NEUMANN."""
        print("Building NEUMANN reasoner...")
        fc, neumann = load_neumann_for_sgg_training(
            lang, rules_to_learn, rules_bk, atoms, self.device, infer_step=4)
        
        neumann.clause_weights = self._softmax_clause_weights(self.rule_weights)
        neumann.print_program()

        V_0 = fc(atoms)
        V_T = neumann(V_0)
        
        target_atoms, target_scores = self._get_target_atoms_with_scores(V_T[0], neumann.atoms)
        self.target_atoms = target_atoms
        return target_atoms, target_scores, V_T, neumann

    def _get_target_atoms_with_scores(self, v_T, atoms, threshold=0.2):
        target_atoms_scores = [(atom, v_T[i]) for i, atom in enumerate(atoms) if atom.pred.name == "target" and v_T[i] > threshold]
        
        if target_atoms_scores:
            return zip(*target_atoms_scores)

        highest_score_index = v_T.argmax()
        return [atoms[highest_score_index]], [v_T[highest_score_index]]

    def segment_objects_by_sam(self, image_source, target_atoms, data_index, image_id):
        """Segment objects by running SAM on extracted crops."""
        boxes_vg = to_boxes(target_atoms, data_index, self.visual_genome_utils_list[0])
        boxes_sgg = to_boxes_with_sgg(target_atoms, image_id, self.visual_genome_utils_list[1])
        boxes = boxes_vg + boxes_sgg

        self.sam_predictor.set_image(image_source)
        
        transformed_boxes = to_transformed_boxes(boxes, image_source, self.sam_predictor, self.device)
        
        try:
            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords=None, point_labels=None, boxes=transformed_boxes, multimask_output=False)
            return masks
        except (RuntimeError, AttributeError):
            print("Error during segmentation")
            return None

    def forward(self, data_index, image_id, scene_graphs, text, image_source):
        """Forwarding function of TrainableDeiSAM."""
        assert len(scene_graphs) == 2, "Currently only 2 SGGs are accepted for learning."

        langs = [
            scene_graph_to_language(scene_graph=scene_graphs[0], text=text, logic_generator=self.llm_logic_generator, num_objects=2),
            get_init_language_with_sgg(scene_graph=scene_graphs[1], text=text, logic_generator=self.llm_logic_generator)
        ]

        llm_rules_list = [self._generate_rules_by_llm(text, lang) for lang in langs]

        scene_graph_atoms_list = []
        scene_graph_langs = []
        
        for i, lang in enumerate(langs):
            if i == 0:
                atoms, scene_lang = self.visual_genome_utils_list[i].data_index_to_atoms(data_index, lang)
            else:
                atoms, scene_lang = self.visual_genome_utils_list[i].image_id_to_atoms(image_id, lang)
            
            scene_graph_atoms_list.append(atoms)
            scene_graph_langs.append(scene_lang)

        sgg_graph_atoms, sgg_lang = translate_atoms_to_sgg_format(scene_graph_atoms_list, scene_graph_langs)
        sgg_rules = translate_rules_to_sgg_format(llm_rules_list)
        
        sgg_target_rules, sgg_lang = get_target_selection_rules(sgg_lang)

        target_atoms, target_scores, V_T, neumann = self.forward_reasoning(sgg_lang, sgg_graph_atoms, sgg_target_rules, sgg_rules)

        if self.sem_uni and torch.cat([s.unsqueeze(-1) for s in target_scores]).max() < 0.1:
            rewritten_rules = self._unify_semantics(langs[0], scene_graph_langs[0], llm_rules_list[0])
            llm_rules_list[1] = rewritten_rules

            sgg_graph_atoms, sgg_lang = translate_atoms_to_sgg_format(scene_graph_atoms_list, scene_graph_langs)
            sgg_rules = translate_rules_to_sgg_format(llm_rules_list)
            
            sgg_target_rules, sgg_lang = get_target_selection_rules(sgg_lang)
            target_atoms, target_scores, V_T, neumann = self.forward_reasoning(sgg_lang, sgg_graph_atoms, sgg_target_rules, sgg_rules)

        masks = self.segment_objects_by_sam(image_source, target_atoms, data_index, image_id) if target_atoms else None
        return masks, target_scores, sgg_rules