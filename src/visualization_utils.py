import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def show_mask(mask, image, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([255 / 255, 10 / 255, 10 / 255, 0.75])
    h, w = mask.shape[-2:]
    if not mask.device.type == "cpu":
        mask = mask.cpu()
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray(
        (mask_image.cpu().numpy() * 255).astype(np.uint8)
    ).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))


def show_mask_with_alpha(mask, image, alpha, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([255 / 255, 10 / 255, 10 / 255, 0.75 * alpha])
    h, w = mask.shape[-2:]
    if not mask.device.type == "cpu":
        mask = mask.cpu()
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray(
        (mask_image.cpu().numpy() * 255).astype(np.uint8)
    ).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))


def get_bbox_by_id(object_id, data_index, vg):
    objects = vg.all_objects[data_index]["objects"]
    target_object = [o for o in objects if int(o["object_id"]) == object_id]
    target_object = target_object[0]
    x = target_object["x"]
    y = target_object["y"]
    w = target_object["w"]
    h = target_object["h"]
    return x, y, w, h


def to_crops(image_source, boxes):
    image_source_crops = []
    for x, y, w, h in boxes:
        image_source_crops = image_source[x : x + w, y : y + h]
    return image_source_crops


def _to_boxes(target_atoms, data_index, vg):
    # return [get_bbox_by_id(id, data_index, vg) for id in object_ids]
    return vg.target_atoms_to_regions(target_atoms, data_index)


def objdata_to_box(data):
    x = data["x"]
    y = data["y"]
    w = data["w"]
    h = data["h"]
    return x, y, w, h


def to_boxes(target_atoms, data_index, vg):
    # get box from relations!! not objects
    object_ids = to_object_ids(target_atoms)
    relations = vg.all_relationships[data_index]["relationships"]
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


def save_box_to_file_GroundedSAM(pr_boxes, gt_boxes, id, dataset, is_short=False):
    if not is_short:
        pr_path = "result/{}/GroundedSAM/prediction/{}.txt".format(dataset, id)
        gt_path = "result/{}/GroundedSAM/ground_truth/{}.txt".format(dataset, id)
    else:
        pr_path = "result_short/visual_genome/prediction/{}.txt".format(id)
        gt_path = "result_short/visual_genome/ground_truth/{}.txt".format(id)
    # save prediction
    text = ""
    for box in pr_boxes:
        text += "target 1.0 {} {} {} {}".format(box[0], box[1], box[2], box[3])
        text += "\n"
    text = text[:-1]
    with open(pr_path, "w") as f:
        f.write(text)

    # save ground truth
    text = ""
    for box in gt_boxes:
        text += "target {} {} {} {}".format(box[0], box[1], box[2], box[3])
        text += "\n"
    text = text[:-1]
    with open(gt_path, "w") as f:
        f.write(text)


def answer_to_boxes(answers):
    if not isinstance(answers, list):
        answer = answers
        x_1 = answer["x"]
        y_1 = answer["y"]
        x_2 = x_1 + answer["w"]
        y_2 = y_1 + answer["h"]
        return [[x_1, y_1, x_2, y_2]]
    else:
        boxes = []
        for answer in answers:
            x_1 = answer["x"]
            y_1 = answer["y"]
            x_2 = x_1 + answer["w"]
            y_2 = y_1 + answer["h"]
            boxes.append([x_1, y_1, x_2, y_2])
        return boxes


def save_box_to_file(pr_boxes, gt_boxes, id, counter, args):
    pr_path = "result/{}_comp{}_learn/{}{}/prediction/{}_vg{}.txt".format(
        args.dataset, args.complexity, args.model, args.sgg_model, counter, id
    )

    gt_path = "result/{}_comp{}_learn/{}{}/ground_truth/{}_vg{}.txt".format(
        args.dataset, args.complexity, args.model, args.sgg_model, counter, id
    )

    # make directories if they do not exist yet
    os.makedirs(
        "result/{}_comp{}/{}{}/prediction".format(
            args.dataset, args.complexity, args.model, args.sgg_model
        ),
        exist_ok=True,
    )
    os.makedirs(
        "result/{}_comp{}/{}{}/ground_truth".format(
            args.dataset, args.complexity, args.model, args.sgg_model
        ),
        exist_ok=True,
    )
    # save prediction
    text = ""
    for box in pr_boxes:
        text += "target 1.0 {} {} {} {}".format(box[0], box[1], box[2], box[3])
        text += "\n"
    text = text[:-1]
    with open(pr_path, "w") as f:
        f.write(text)

    # save ground truth
    text = ""
    for box in gt_boxes:
        text += "target {} {} {} {}".format(box[0], box[1], box[2], box[3])
        text += "\n"
    text = text[:-1]
    with open(gt_path, "w") as f:
        f.write(text)


def save_segmented_images(
    counter,
    vg_image_id,
    annotated_frame_with_mask,
    data_index,
    deictic_representation,
    base_path="imgs/",
):
    im = Image.fromarray(annotated_frame_with_mask)
    rgb_im = im.convert("RGB")
    # rgb_im.save("imgs/{}_sam.jpeg".format(data_index))

    # caption = deictic_representation

    deictic_representation = deictic_representation.replace("/", " ").replace(".", "")

    save_path = base_path + "deicticVG_ID:{}_VGImID:{}_{}.png".format(
        counter, vg_image_id, deictic_representation
    )
    rgb_im.save(save_path)

    # plt.figure()
    # plt.imshow(rgb_im)
    # plt.axis("off")
    # plt.xlabel(caption)
    # plt.tight_layout()
    # plt.savefig(
    #     base_path
    #     + "deicticVG_ID:{}_VGImID:{}_{}".format(
    #         counter, vg_image_id, deictic_representation
    #     ),
    #     bbox_inches="tight",
    #     pad_inches=0,
    # )
    print("A figure saved to {}".format(save_path))


def save_segmented_images_with_target_scores(
    counter,
    vg_image_id,
    annotated_frame_with_mask,
    data_index,
    deictic_representation,
    mask_probs,
    base_path="imgs/",
):
    im = Image.fromarray(annotated_frame_with_mask)
    rgb_im = im.convert("RGB")
    # rgb_im.save("imgs/{}_sam.jpeg".format(data_index))

    # caption = deictic_representation

    deictic_representation = deictic_representation.replace("/", " ").replace(".", "")

    if len(mask_probs) > 3:
        mask_probs = mask_probs[:3]
    scores_str = str(np.round(np.array(mask_probs), 2).tolist())
    save_path = base_path + "deicticVG_ID:{}_VGImID:{}_{}_scores_{}.png".format(
        counter,
        vg_image_id,
        deictic_representation,
        scores_str,
    )
    rgb_im.save(save_path)
    print("A figure saved to {}".format(save_path))
