import os
from PIL import Image
import numpy as np
import torch
from sam_utils import to_object_ids


def apply_random_color():
    return np.concatenate([np.random.random(3), np.array([0.8])], axis=0)


def apply_default_color(alpha=0.75):
    return np.array([255 / 255, 10 / 255, 10 / 255, alpha])


def get_colored_mask(mask, color):
    h, w = mask.shape[-2:]
    mask = mask.cpu() if mask.device.type != "cpu" else mask
    return mask.reshape(h, w, 1) * color.reshape(1, 1, -1)


def overlay_mask_on_image(mask, image, color):
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray(
        (mask.cpu().numpy() * 255).astype(np.uint8)
    ).convert("RGBA")
    return Image.alpha_composite(annotated_frame_pil, mask_image_pil)


def show_mask(mask, image, random_color=False):
    color = apply_random_color() if random_color else apply_default_color()
    mask_image = get_colored_mask(mask, color)
    return np.array(overlay_mask_on_image(mask_image, image, color))


def show_mask_with_alpha(mask, image, alpha, random_color=False):
    color = apply_random_color() if random_color else apply_default_color(alpha * 0.75)
    mask_image = get_colored_mask(mask, color)
    return np.array(overlay_mask_on_image(mask_image, image, color))


def get_bbox_by_id(object_id, data_index, vg):
    objects = vg.all_objects[data_index]["objects"]
    target_object = next((o for o in objects if int(o["object_id"]) == object_id), None)
    
    if not target_object:
        raise ValueError(f"Object with ID {object_id} not found.")
    
    return target_object["x"], target_object["y"], target_object["w"], target_object["h"]


def to_crops(image_source, boxes):
    return [image_source[x:x + w, y:y + h] for x, y, w, h in boxes]


def _to_boxes(target_atoms, data_index, vg):
    return vg.target_atoms_to_regions(target_atoms, data_index)


def objdata_to_box(data):
    return data["x"], data["y"], data["w"], data["h"]


def to_boxes(target_atoms, data_index, vg):
    object_ids = to_object_ids(target_atoms)
    relations = vg.all_relationships[data_index]["relationships"]
    
    return [
        objdata_to_box(rel["object"] if rel["object"]["object_id"] == id else rel["subject"])
        for id in object_ids
        for rel in relations
        if rel["object"]["object_id"] == id or rel["subject"]["object_id"] == id
    ]

# def to_boxes(target_atoms, data_index, vg):
#     # get box from relations!! not objects
#     object_ids = to_object_ids(target_atoms)
#     relations = vg.all_relationships[data_index]["relationships"]
#     boxes = []
#     for id in object_ids:
#         for rel in relations:
#             if rel["object"]["object_id"] == id:
#                 boxes.append(objdata_to_box(rel["object"]))
#                 break
#             elif rel["subject"]["object_id"] == id:
#                 boxes.append(objdata_to_box(rel["subject"]))
#                 break
#     return boxes

def to_xyxy(boxes):
    xyxy_boxes = [torch.tensor([x, y, x + w, y + h]) for x, y, w, h in boxes]
    return torch.stack(xyxy_boxes)


def save_boxes_to_file(boxes, path, is_prediction=True):
    text = "\n".join(f"target 1.0 {box[0]} {box[1]} {box[2]} {box[3]}" for box in boxes)
    with open(path, "w") as f:
        f.write(text)
    print(f"File saved to {path}")


def save_box_to_file(pr_boxes, gt_boxes, id, counter, args):
    dirs = [
        f"result/{args.dataset}_comp{args.complexity}/{args.model}/prediction",
        f"result/{args.dataset}_comp{args.complexity}/{args.model}/ground_truth",
    ]

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

    pr_path = f"{dirs[0]}/{counter}_vg{id}.txt"
    gt_path = f"{dirs[1]}/{counter}_vg{id}.txt"

    save_boxes_to_file(pr_boxes, pr_path)
    save_boxes_to_file(gt_boxes, gt_path, is_prediction=False)


def answer_to_boxes(answers):
    if not isinstance(answers, list):
        answers = [answers]
    
    return [[answer["x"], answer["y"], answer["x"] + answer["w"], answer["y"] + answer["h"]] for answer in answers]


def save_segmented_images(counter, vg_image_id, image_with_mask, description, base_path="imgs/"):
    image_with_mask = Image.fromarray(image_with_mask).convert("RGB")
    description = description.replace("/", " ").replace(".", "")
    save_path = f"{base_path}deicticVG_ID:{counter}_VGImID:{vg_image_id}_{description}.png"
    image_with_mask.save(save_path)
    print(f"Image saved to {save_path}")


def save_segmented_images_with_target_scores(counter, vg_image_id, image_with_mask, description, target_scores, base_path="imgs/"):
    if len(target_scores) > 3:
        target_scores = target_scores[:3]
    scores_str = str(np.round(target_scores, 2).tolist())
    save_segmented_images(counter, vg_image_id, image_with_mask, f"{description}_scores_{scores_str}", base_path)