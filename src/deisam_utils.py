import os
import random

import cv2
import torch
from PIL import Image
from segment_anything import SamPredictor, build_sam
from torchvision.ops import masks_to_boxes
from huggingface_hub import hf_hub_download

# Visualization and utility imports
from groundingdino.util.inference import annotate
from visualization_utils import (
    answer_to_boxes,
    save_box_to_file,
    save_segmented_images,
    save_segmented_images_with_target_scores,
    show_mask,
    show_mask_with_alpha,
)

# Neumann library imports
from neumann.facts_converter import FactsConverter
from neumann.fol.data_utils import DataUtils
from neumann.fol.language import DataType
from neumann.fol.logic import Atom, Const, Predicate
from neumann.logic_utils import generate_atoms
from neumann.neumann_utils import get_neumann_model, get_trainable_neumann_model


def load_neumann(lang, rules, atoms, device, infer_step=2):
    """Load a Neumann reasoner model with specified language, rules, and atoms."""
    atoms = add_target_cond_atoms(lang, atoms)
    fc = FactsConverter(lang, atoms, [], device)
    reasoner = get_neumann_model(
        rules, [], [], atoms, lang.consts, lang, 1, infer_step, device
    )
    return fc, reasoner


def load_neumann_for_sgg_training(lang, rules_to_learn, rules_bk, atoms, device, infer_step=4):
    """Load a trainable Neumann model for scene graph generation (SGG) training."""
    atoms = add_target_cond_atoms_for_sgg_training(lang, atoms)
    fc = FactsConverter(lang, atoms, [], device)
    reasoner = get_trainable_neumann_model(
        rules_to_learn, rules_bk, [], atoms, lang.consts, lang, 1, infer_step, device
    )
    return fc, reasoner


def load_sam_model(device):
    """Load SAM model from a checkpoint file."""
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.to(device=device)
    return SamPredictor(sam)


def crop_objects(img, masks):
    """Crop objects from the image using provided masks."""
    cropped_objects = []

    for mask in masks:
        x, y, width, height = mask["bbox"]

        if width * height > 2000:
            cropped_image = img[int(y):int(y + height), int(x):int(x + width), :]
            cropped_objects.append(cropped_image)

    return cropped_objects


def load_image(path):
    """Load an image from the specified file path and convert it to RGB."""
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def extract_target_ids(v_T, threshold):
    """Extract target IDs from a tensor of valuations using a threshold."""
    target_ids = []
    obj_id = 0

    for atom in self.reasoner.atoms:
        if atom.pred.name == "target":
            index = self.reasoner.atoms.index(atom)
            if v_T[index] > threshold:
                target_ids.append(obj_id)
            obj_id += 1

    return target_ids


def load_model_from_hf(repo_id, filename, config_filename, device="cpu"):
    """Load a model from Hugging Face Hub."""
    config_path = hf_hub_download(repo_id=repo_id, filename=config_filename)
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict

    args = SLConfig.fromfile(config_path)
    model = build_model(args)
    args.device = device

    model_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(model_file, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(f"Model loaded from {model_file}")

    return model.eval()


def segment_by_grounded_sam(self, image, image_source, text_prompt, box_threshold=0.35, text_threshold=0.25):
    """Segment an image using a grounded SAM model."""
    boxes, logits, phrases = predict(
        model=self.model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    annotated_frame = annotate(
        image_source=image_source, boxes=boxes, logits=logits, phrases=phrases
    )[..., ::-1]  # Convert BGR to RGB
    return boxes, logits, phrases, annotated_frame


def save_box_results(args, masks, answer, file_id, counter):
    """Save bounding boxes of predicted segmentations with ground truth."""
    pr_boxes = masks_to_boxes(masks.squeeze(1).to(torch.int32))
    gt_boxes = answer_to_boxes(answer)
    save_box_to_file(pr_boxes, gt_boxes, file_id, counter, args)
    return pr_boxes, gt_boxes


def save_segmentation_result(args, masks, answer, image_source, counter, vg_image_id, data_index, deictic_representation, is_failed):
    """Plot and save segmentation masks on original visual inputs."""
    annotated_frame = image_source

    for mask in masks:
        annotated_frame = show_mask(mask[0], annotated_frame)

    base_path = f"plot/{args.dataset}_comp{args.complexity}/DeiSAM{args.sgg_model}/"
    os.makedirs(base_path, exist_ok=True)

    save_segmented_images(
        counter=counter,
        vg_image_id=vg_image_id,
        image_with_mask=annotated_frame,
        description=deictic_representation,
        base_path=base_path,
    )

    if is_failed:
        failed_path = f"plot/{args.dataset}_comp{args.complexity}/DeiSAM{args.sgg_model}_failed/"
        os.makedirs(failed_path, exist_ok=True)
        save_segmented_images(
            counter=counter,
            vg_image_id=vg_image_id,
            annotated_frame_with_mask=annotated_frame,
            description=deictic_representation,
            base_path=failed_path,
        )


def save_segmentation_result_with_alphas(args, masks, mask_probs, answer, image_source, counter, vg_image_id, data_index, deictic_representation, iter):
    """Plot and save segmentation masks with alpha values on visual inputs."""
    annotated_frame = image_source

    for i, mask in enumerate(masks):
        annotated_frame = show_mask_with_alpha(mask=mask[0], image=annotated_frame, alpha=mask_probs[i])

    base_path = f"learning_plot/{args.dataset}_comp{args.complexity}/seed_{args.seed}/iter_{iter}/DeiSAM{args.sgg_model}/"
    os.makedirs(base_path, exist_ok=True)

    save_segmented_images_with_target_scores(
        counter=counter,
        vg_image_id=vg_image_id,
        image_with_mask=annotated_frame,
        deictic_representation=deictic_representation,
        mask_probs=mask_probs,
        base_path=base_path,
    )


def save_llm_response(args, pred_response, rule_response, counter, image_id, deictic_representation):
    """Save LLM responses (pred & rule) to files."""
    base_path = f"llm_output/{args.dataset}_comp{args.complexity}/DeiSAM{args.sgg_model}/"
    os.makedirs(f"{base_path}pred_response/", exist_ok=True)
    os.makedirs(f"{base_path}rule_response/", exist_ok=True)

    pred_file = f"{counter}_vg{image_id}_{deictic_representation.replace('.', '').replace('/', ' ')}.txt"
    rule_file = f"{counter}_vg{image_id}_{deictic_representation.replace('.', '').replace('/', ' ')}.txt"

    with open(f"{base_path}pred_response/{pred_file}", "w") as f:
        f.write(pred_response)
    
    with open(f"{base_path}rule_response/{rule_file}", "w") as f:
        f.write(rule_response)


def get_random_masks(model):
    """Get random masks from the Neumann model's target atoms."""
    targets = [atom for atom in model.neumann.atoms if "target(obj_" in str(atom)]
    return [random.choice(targets)]


def add_target_cond_atoms(lang, atoms):
    """Add target and condition atoms to the atoms list."""
    spec_predicate = Predicate(".", 1, [DataType("spec")])
    true_atom = Atom(spec_predicate, [Const("__T__", dtype=DataType("spec"))])

    target_atoms = generate_target_atoms(lang)
    cond_atoms = generate_cond_atoms(lang)
    return [true_atom] + sorted(set(atoms)) + target_atoms + cond_atoms


def generate_target_atoms(lang):
    """Generate target atoms for each object constant in the language."""
    target_predicate = Predicate("target", 1, [DataType("object")])
    return [
        Atom(target_predicate, [const])
        for const in lang.consts if const.dtype.name == "object" and "obj_" in const.name
    ]




def generate_cond_atoms(lang):
    """Generate condition atoms for each object constant in the language."""
    cond_predicates = [Predicate(f"cond{i}", 1, [DataType("object")]) for i in range(1, 4)]
    cond_atoms = []
    for pred in cond_predicates:
        cond_atoms.extend(
            Atom(pred, [const])
            for const in lang.consts if const.dtype.name == "object" and "obj_" in const.name
        )
    return sorted(set(cond_atoms))


def add_target_cond_atoms_for_sgg_training(lang, atoms, num_sgg_models=2):
    """Add target and condition atoms for SGG training to the list of atoms."""
    spec_predicate = Predicate(".", 1, [DataType("spec")])
    true_atom = Atom(spec_predicate, [Const("__T__", dtype=DataType("spec"))])

    target_atoms = generate_target_atoms_for_sgg_training(lang, num_sgg_models)
    cond_atoms = generate_cond_atoms_for_sgg_training(lang, num_sgg_models)
    return [true_atom] + sorted(set(atoms)) + target_atoms + cond_atoms


def generate_target_atoms_for_sgg_training(lang, num_sgg_models):
    """Generate target atoms for main and SGG models."""
    main_target_pred = Predicate("target", 1, [DataType("object")])
    sgg_target_preds = [
        Predicate(f"target_sgg{i}", 1, [DataType("object")]) for i in range(num_sgg_models)
    ]

    all_target_preds = [main_target_pred] + sgg_target_preds
    target_atoms = [
        Atom(pred, [const])
        for pred in all_target_preds
        for const in lang.consts if const.dtype.name == "object" and "obj_" in const.name
    ]

    return sorted(set(target_atoms))


def generate_cond_atoms_for_sgg_training(lang, num_sgg_models):
    """Generate conditional atoms for SGG training."""
    cond_atoms = []
    
    for i in range(num_sgg_models):
        cond_preds = [Predicate(f"cond{j}_sgg{i}", 1, [DataType("object")]) for j in range(1, 4)]
        
        for cond_pred in cond_preds:
            cond_atoms.extend(
                Atom(cond_pred, [const])
                for const in lang.consts if const.dtype.name == "object" and "obj_" in const.name
            )

    return sorted(set(cond_atoms))