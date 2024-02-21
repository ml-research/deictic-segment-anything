import torch
from groundingdino.util import box_ops
from itsdangerous import NoneAlgorithm
from visual_genome_utils import objdata_to_box, scene_graph_to_language


def to_xyxy(boxes):
    """xywh to xyxy format

    Args:
        boxes (_type_): _description_

    Returns:
        _type_: _description_
    """
    xyxy_boxes = []
    for box in boxes:
        x = box[0]
        y = box[1]
        # x + w
        x_2 = x + box[2]
        # y + h
        y_2 = y + box[3]
        xyxy_boxes.append(torch.tensor([x, y, x_2, y_2]))
    return torch.stack(xyxy_boxes)


def transform_boxes(boxes, image_source):
    # box: normalized box xywh -> unnormalized xyxy
    H, W, _ = image_source.shape
    # boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    try:
        boxes_xyxy = to_xyxy(boxes)  # * torch.Tensor([W, H, W, H])
        return boxes_xyxy
    except RuntimeError:
        return NoneAlgorithm


def to_object_ids(target_atoms):
    object_ids = [int(atom.terms[0].name.split("_")[-1]) for atom in target_atoms]
    return object_ids


def to_boxes(target_atoms, data_index, visual_genome_utils):
    # get box from relations!! not objects
    object_ids = to_object_ids(target_atoms)
    relations = visual_genome_utils.all_relationships[data_index]["relationships"]
    boxes = []
    for id in object_ids:
        for rel in relations:
            if rel["object"]["object_id"] == id:
                boxes.append(objdata_to_box(rel["object"]))
                break
            elif rel["subject"]["object_id"] == id:
                boxes.append(objdata_to_box(rel["subject"]))
                break
    return boxes


def to_boxes_with_sgg(target_atoms, image_id, visual_genome_utils):
    # get box from relations!! not objects
    object_ids = to_object_ids(target_atoms)
    relations = visual_genome_utils.all_relationships[image_id]
    boxes = []
    for id in object_ids:
        for rel in relations:
            if rel["o_unique_id"] == id:
                boxes.append(rel["obox"])
                break
            if rel["s_unique_id"] == id:
                boxes.append(rel["sbox"])
                break
    return boxes


def to_transformed_boxes(boxes, image_source, sam_predictor, device):
    # box: normalized box xywh -> unnormalized xyxy
    H, W, _ = image_source.shape
    # boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    try:
        boxes_xyxy = to_xyxy(boxes)  # * torch.Tensor([W, H, W, H])
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(
            boxes_xyxy, image_source.shape[:2]
        ).to(device)
        return transformed_boxes
    except RuntimeError:
        return []
