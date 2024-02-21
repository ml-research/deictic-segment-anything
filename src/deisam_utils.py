import os
import random

import cv2
import matplotlib.pyplot as plt
import torch
from groundingdino.util.inference import annotate
from huggingface_hub import hf_hub_download
from PIL import Image
from segment_anything import SamPredictor, build_sam
from torchvision.ops import masks_to_boxes
from visualization_utils import (
    answer_to_boxes,
    save_box_to_file,
    save_segmented_images,
    save_segmented_images_with_target_scores,
    show_mask,
    show_mask_with_alpha,
)

from neumann.facts_converter import FactsConverter
from neumann.fol.data_utils import DataUtils
from neumann.fol.language import DataType
from neumann.fol.logic import Atom, Const, Predicate
from neumann.logic_utils import generate_atoms  # , get_neumann_atoms
from neumann.neumann_utils import get_neumann_model, get_trainable_neumann_model


def load_neumann(lang, rules, atoms, device, infer_step=2):
    # TODO: maybe here we add the pruning of terms by parsing atoms and collect only needed object/attribute terms
    terms = lang.consts
    bk = []
    bk_clauses = []
    term_depth = 1
    infer_step = infer_step
    # atoms = generate_atoms(lang, terms)
    atoms = add_target_cond_atoms(lang, atoms)
    fc = FactsConverter(lang, atoms, bk, device)
    reasoner = get_neumann_model(
        rules,
        bk_clauses,
        bk,
        atoms,
        terms,
        lang,
        term_depth,
        infer_step,
        device,
    )
    return fc, reasoner


def load_neumann_for_sgg_training(
    lang, rules_to_learn, rules_bk, atoms, device, infer_step=4
):
    # TODO: maybe here we add the pruning of terms by parsing atoms and collect only needed object/attribute terms
    terms = lang.consts
    bk = []
    bk_clauses = rules_bk
    term_depth = 1
    infer_step = infer_step
    # atoms = generate_atoms(lang, terms)
    atoms = add_target_cond_atoms_for_sgg_training(lang, atoms)
    fc = FactsConverter(lang, atoms, bk, device)
    # reasoner = get_trainable_neumann_model(
    reasoner = get_neumann_model(
        rules_to_learn,
        bk_clauses,
        bk,
        atoms,
        terms,
        lang,
        term_depth,
        infer_step,
        device,
    )
    return fc, reasoner


def load_sam_model(device):
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    return sam_predictor


def crop_objects(img, masks):
    crop_boxes = [mask["bbox"] for mask in masks]

    cropped_objects = []
    for crop_box in crop_boxes:
        # x_1 = crop_box[0]
        # y_1 = crop_box[1]
        # x_2 = x_1 + crop_box[2]
        # y_2 = y_1 + crop_box[3]
        # x_1, y_1, x_2, y_2 = xywh2xyxy(*crop_box)
        # obj = img[x_1:x_2, y_1:y_2]
        x, y, width, height = crop_box
        cropped_image = img[int(y) : int(y + height), int(x) : int(x + width), :]
        # filename = os.path.join(output_directory, str(i) + '.png')
        # cv2.imwrite(filename, cropped_image)
        if width * height > 2000:
            cropped_objects.append(cropped_image)
    return cropped_objects


def load_image(path):
    image = cv2.imread(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def extract_target_ids(v_T, th):
    target_ids = []
    obj_id = 0
    atoms = self.reasoner.atoms
    for atom in atoms:
        if atom.pred.name == "target":
            index = atoms.index(atom)
            prob = v_T[index]
            if prob > th:
                target_ids.append(obj_id)
            obj_id += 1
    return target_ids


def load_model_hf(repo_id, filename, ckpt_config_filename, device="cpu"):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict

    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location="cpu")
    log = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model


def segment_by_grounded_sam(
    self,
    image,
    image_source,
    text_prompt,
    box_threashold=0.35,
    text_threashold=0.25,
):
    boxes, logits, phrases = predict(
        model=self.model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threashold,
        text_threshold=text_threashold,
    )
    annotated_frame = annotate(
        image_source=image_source, boxes=boxes, logits=logits, phrases=phrases
    )
    annotated_frame = annotated_frame[..., ::-1]  # BGR to RGB
    return boxes, logits, phrases, annotated_frame


def save_box_results(args, masks, answer, id, counter):
    """Save bounding boxes of predicted segmentations together wieh ground truth."""
    # save the result
    pr_boxes = masks_to_boxes(masks.squeeze(1).to(torch.int32))
    gt_boxes = answer_to_boxes(answer)
    save_box_to_file(pr_boxes, gt_boxes, id, counter, args)
    return pr_boxes, gt_boxes


def save_segmentation_result(
    args,
    masks,
    answer,
    image_source,
    counter,
    vg_image_id,
    data_index,
    deictic_representation,
    is_failed,
):
    """Plot and save segmentation masks on original visual inputs."""
    logits = [1.0 for m in masks]
    phrases = ["" for m in masks]
    # boxes = torch.stack([torch.tensor(b) for b in boxes])
    # annotated_frame = annotate(
    #     image_source=image_source, boxes=boxes, logits=logits, phrases=phrases
    # )
    # annotated_frame = annotated_frame[..., ::-1]  # BGR to RGB
    # annotated_frame_with_mask = annotated_frame
    annotated_frame_with_mask = image_source
    for i in range(len(masks)):
        # try:
        annotated_frame_with_mask = show_mask(masks[i][0], annotated_frame_with_mask)
        # except ValueError:
        #     next

    # make directoris to save images with masks
    os.makedirs(
        "plot/{}_comp{}/DeiSAM{}/".format(
            args.dataset, args.complexity, args.sgg_model
        ),
        exist_ok=True,
    )
    os.makedirs(
        "plot/{}_comp{}/DeiSAM{}_failed/".format(
            args.dataset, args.complexity, args.sgg_model
        ),
        exist_ok=True,
    )

    save_segmented_images(
        counter=counter,
        vg_image_id=vg_image_id,
        annotated_frame_with_mask=annotated_frame_with_mask,
        data_index=data_index,
        deictic_representation=deictic_representation,
        base_path="plot/{}_comp{}/DeiSAM{}/".format(
            args.dataset, args.complexity, args.sgg_model
        ),
    )
    # save failed cases
    if is_failed:
        save_segmented_images(
            counter=counter,
            vg_image_id=vg_image_id,
            annotated_frame_with_mask=annotated_frame_with_mask,
            data_index=data_index,
            deictic_representation=deictic_representation,
            base_path="plot/{}_comp{}/DeiSAM{}_failed/".format(
                args.dataset, args.complexity, args.sgg_model
            ),
        )


def save_segmentation_result_with_alphas(
    args,
    masks,
    mask_probs,
    answer,
    image_source,
    counter,
    vg_image_id,
    data_index,
    deictic_representation,
    iter,
):
    """Plot and save segmentation masks on original visual inputs."""
    # logits = [1.0 for m in masks]
    # phrases = ["" for m in masks]
    # boxes = torch.stack([torch.tensor(b) for b in boxes])
    # annotated_frame = annotate(
    #     image_source=image_source, boxes=boxes, logits=logits, phrases=phrases
    # )
    # annotated_frame = annotated_frame[..., ::-1]  # BGR to RGB
    # annotated_frame_with_mask = annotated_frame
    annotated_frame_with_mask = image_source
    for i in range(len(masks)):
        # try:
        annotated_frame_with_mask = show_mask_with_alpha(
            mask=masks[i][0],
            image=annotated_frame_with_mask,
            alpha=mask_probs[i],
        )
        # except ValueError:
        #     next

    # make directoris to save images with masks
    os.makedirs(
        "learning_plot/{}_comp{}/seed_{}/iter_{}/DeiSAM{}/".format(
            args.dataset, args.complexity, args.seed, iter, args.sgg_model
        ),
        exist_ok=True,
    )
    save_segmented_images_with_target_scores(
        counter=counter,
        vg_image_id=vg_image_id,
        annotated_frame_with_mask=annotated_frame_with_mask,
        data_index=data_index,
        deictic_representation=deictic_representation,
        mask_probs=mask_probs,
        base_path="learning_plot/{}_comp{}/seed_{}/iter_{}/DeiSAM{}/".format(
            args.dataset, args.complexity, args.seed, iter, args.sgg_model
        ),
    )
    # # save failed cases
    # if is_failed:
    #     save_segmented_images(
    #         counter=counter,
    #         vg_image_id=vg_image_id,
    #         annotated_frame_with_mask=annotated_frame_with_mask,
    #         data_index=data_index,
    #         deictic_representation=deictic_representation,
    #         base_path="plot/{}_comp{}/DeiSAM{}_failed/".format(
    #             args.dataset, args.complexity, args.sgg_model
    #         ),
    #     )


def save_llm_response(
    args, pred_response, rule_response, counter, image_id, deictic_representation
):
    base_path = "llm_output/{}_comp{}/DeiSAM{}/".format(
        args.dataset, args.complexity, args.sgg_model
    )
    os.makedirs(base_path + "pred_response/", exist_ok=True)
    os.makedirs(base_path + "rule_response/", exist_ok=True)

    pred_response_file = "{}_vg{}_{}.txt".format(
        counter, image_id, deictic_representation.replace(".", "").replace("/", " ")
    )

    with open(base_path + "pred_response/" + pred_response_file, "w") as f:
        f.write(pred_response)

    rule_response_file = "{}_vg{}_{}.txt".format(
        counter, image_id, deictic_representation.replace(".", "").replace("/", " ")
    )

    with open(base_path + "rule_response/" + rule_response_file, "w") as f:
        f.write(rule_response)


def get_random_masks(model):
    targets = [atom for atom in model.neumann.atoms if "target(obj_" in str(atom)]
    target_atoms = [random.choice(targets)]
    return target_atoms


def add_target_cond_atoms(lang, atoms):
    p_ = Predicate(".", 1, [DataType("spec")])
    # false = Atom(p_, [Const("__F__", dtype=DataType("spec"))])
    true = Atom(p_, [Const("__T__", dtype=DataType("spec"))])

    target_atoms = generate_target_atoms(lang)
    cond_atoms = generate_cond_atoms(lang)
    return [true] + sorted(list(set(atoms))) + target_atoms + cond_atoms


def generate_target_atoms(lang):
    target_atoms = []
    # add target predicates and atoms
    p_target = Predicate("target", 1, [DataType("object")])
    for const in lang.consts:
        if const.dtype.name == "object" and "obj_" in const.name:
            # add e.g. target(obj_1234)
            target_atom = Atom(p_target, [const])
            target_atoms.append(target_atom)
    return sorted(list(set(target_atoms)))


def generate_cond_atoms(lang):
    cond_atoms = []
    # add condition predicates and atoms
    p_cond1 = Predicate("cond1", 1, [DataType("object")])
    p_cond2 = Predicate("cond2", 1, [DataType("object")])
    p_cond3 = Predicate("cond3", 1, [DataType("object")])
    for const in lang.consts:
        if const.dtype.name == "object" and "obj_" in const.name:
            # add e.g. cond1(obj_1234)
            cond1_atom = Atom(p_cond1, [const])
            cond2_atom = Atom(p_cond2, [const])
            cond3_atom = Atom(p_cond3, [const])
            cond_atoms.extend([cond1_atom, cond2_atom, cond3_atom])
    return sorted(list(set(cond_atoms)))


def add_target_cond_atoms_for_sgg_training(lang, atoms, num_sgg_models=2):
    p_ = Predicate(".", 1, [DataType("spec")])
    # false = Atom(p_, [Const("__F__", dtype=DataType("spec"))])
    true = Atom(p_, [Const("__T__", dtype=DataType("spec"))])

    target_atoms = generate_target_atoms_for_sgg_training(lang, num_sgg_models)
    cond_atoms = generate_cond_atoms_for_sgg_training(lang, num_sgg_models)
    return [true] + sorted(list(set(atoms))) + target_atoms + cond_atoms


def generate_target_atoms_for_sgg_training(lang, num_sgg_models):
    """generate target(obj1), target_sgg0(obj1), ..."""
    target_atoms = []
    # add target predicates and atoms
    p_target_main = Predicate("target", 1, [DataType("object")])
    p_target_sgg_list = [
        Predicate("target_sgg{}".format(i), 1, [DataType("object")])
        for i in range(num_sgg_models)
    ]
    p_target_list = [p_target_main] + p_target_sgg_list
    for p_target in p_target_list:
        for const in lang.consts:
            if const.dtype.name == "object" and "obj_" in const.name:
                # add e.g. target(obj_1234)
                target_atom = Atom(p_target, [const])
                target_atoms.append(target_atom)
    return sorted(list(set(target_atoms)))


def generate_cond_atoms_for_sgg_training(lang, num_sgg_models):
    """generate cond1(obj1), cond1_sgg0(obj1), ..."""
    cond_atoms = []
    # add condition predicates and atoms
    for i in range(num_sgg_models):
        p_cond1 = Predicate("cond1_sgg{}".format(i), 1, [DataType("object")])
        p_cond2 = Predicate("cond2_sgg{}".format(i), 1, [DataType("object")])
        p_cond3 = Predicate("cond3_sgg{}".format(i), 1, [DataType("object")])
        for const in lang.consts:
            if const.dtype.name == "object" and "obj_" in const.name:
                # add e.g. cond1(obj_1234)
                cond1_atom = Atom(p_cond1, [const])
                cond2_atom = Atom(p_cond2, [const])
                cond3_atom = Atom(p_cond3, [const])
                cond_atoms.extend([cond1_atom, cond2_atom, cond3_atom])
    return sorted(list(set(cond_atoms)))
