import torch
from groundingdino.util import box_ops
from visual_genome_utils import objdata_to_box


def to_xyxy(boxes):
    """Convert boxes from xywh format to xyxy format.

    Args:
        boxes (torch.Tensor): A tensor of boxes in xywh format.

    Returns:
        torch.Tensor: A tensor of boxes in xyxy format.
    """
    xyxy_boxes = [
        torch.tensor([x, y, x + w, y + h])
        for x, y, w, h in boxes
    ]
    return torch.stack(xyxy_boxes)


def transform_boxes(boxes, image_source):
    """Transform normalized xywh boxes to unnormalized xyxy boxes.

    Args:
        boxes (torch.Tensor): A tensor of boxes in xywh format.
        image_source (numpy.ndarray): The source image to obtain dimensions.

    Returns:
        torch.Tensor or NoneAlgorithm: Transformed boxes in xyxy format, or NoneAlgorithm on failure.
    """
    try:
        return to_xyxy(boxes)
    except RuntimeError:
        return NoneAlgorithm


def to_object_ids(target_atoms):
    """Extract object IDs from FOL target atoms.

    Args:
        target_atoms (list): List of target atoms.

    Returns:
        list: A list of object IDs.
    """
    return [int(atom.terms[0].name.split("_")[-1]) for atom in target_atoms]


def to_boxes(target_atoms, data_index, visual_genome_utils):
    """Extract bounding boxes for target atoms using Visual Genome data.

    Args:
        target_atoms (list): List of target atoms.
        data_index (int): Index in the Visual Genome dataset.
        visual_genome_utils (object): Utilities for accessing Visual Genome data.

    Returns:
        list: A list of bounding boxes.
    """
    object_ids = to_object_ids(target_atoms)
    relations = visual_genome_utils.all_relationships[data_index]["relationships"]

    boxes = []
    for obj_id in object_ids:
        for rel in relations:
            if rel["object"]["object_id"] == obj_id:
                boxes.append(objdata_to_box(rel["object"]))
                break
            elif rel["subject"]["object_id"] == obj_id:
                boxes.append(objdata_to_box(rel["subject"]))
                break
    return boxes


def to_boxes_with_sgg(target_atoms, image_id, visual_genome_utils):
    """Extract bounding boxes for target atoms using Scene Graph Generation data.

    Args:
        target_atoms (list): List of target atoms.
        image_id (int): Image ID for accessing relationships.
        visual_genome_utils (object): Utilities for accessing Visual Genome data.

    Returns:
        list: A list of bounding boxes.
    """
    object_ids = to_object_ids(target_atoms)
    relations = visual_genome_utils.all_relationships[image_id]

    boxes = []
    for obj_id in object_ids:
        for rel in relations:
            if rel["o_unique_id"] == obj_id:
                boxes.append(rel["obox"])
                break
            if rel["s_unique_id"] == obj_id:
                boxes.append(rel["sbox"])
                break
    return boxes


def to_transformed_boxes(boxes, image_source, sam_predictor, device):
    """Apply transformations to bounding boxes using a SAM predictor.

    Args:
        boxes (list): List of bounding boxes.
        image_source (numpy.ndarray): The source image.
        sam_predictor (object): SAM predictor object with transform capabilities.
        device (torch.device): The device to convert the boxes to.

    Returns:
        torch.Tensor: A tensor of transformed boxes.
    """
    try:
        boxes_xyxy = to_xyxy(boxes)
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(
            boxes_xyxy, image_source.shape[:2]
        ).to(device)
        return transformed_boxes
    except RuntimeError:
        return []