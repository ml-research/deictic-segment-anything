import argparse
import os
import random
import time

import lark
import numpy as np
import openai
import rtpt as RTPT
import torch
from data_vg import (
    DeicticVisualGenome,
    DeicticVisualGenomeSGGTraining,
    PredictedSceneGraphUtils,
    VisualGenomeUtils,
)
from deisam import DeiSAM, DeiSAMSGG, TrainableDeiSAM
from deisam_utils import (
    get_random_masks,
    save_box_results,
    save_segmentation_result_with_alphas,
)
from groundingdino.util.inference import annotate
from hypothesis import target
from learning_utils import are_all_targets_detected, to_bce_examples
from regex import P

# from learning_utils import get_target_selection_rules, translate_rules_to_sgg_format
from rtpt import RTPT
from sklearn.metrics import accuracy_score, recall_score, roc_curve
from torchvision.ops import masks_to_boxes
from visualization_utils import answer_to_boxes, save_box_to_file, to_xyxy

torch.set_num_threads(10)

from torchvision.ops import generalized_box_iou_loss


def test_deisam_sgg(
    args,
    deisam,
    data_loader,
    val_data_loader,
    test_data_loader,
    vg_1,
    vg_2,
    epochs,
    start_id,
    end_id,
    device,
):
    steps = 1
    rtpt = RTPT(
        name_initials="",
        experiment_name="TestSGGDeiSAM{}".format(args.complexity),
        max_iterations=1,
    )
    rtpt.start()
    # setup the optimizer
    params = list(deisam.parameters())

    # generate log folder
    os.makedirs("learning_logs/comp{}".format(args.complexity), exist_ok=True)

    deisam.rule_weights = torch.nn.Parameter(
        torch.nn.Parameter(torch.tensor([[-100.0, 100.0]]).to(device))
    )
    counter = 0
    # copute the initial test score
    test_acc_path = (
        # "learning_logs/comp{}/test_acc_iter_{}_seed_{}.txt".format(
        "learning_logs/comp{}/test_acc_iter_{}_sgg_seed_{}.txt".format(
            args.complexity, counter, args.seed
        )
    )
    print("Computing Test Accuracy... on Iter {}".format(counter))
    test_acc, test_rec = compute_test_accuracy(
        args, deisam, vg_1, vg_2, test_data_loader, counter, n_data=400
    )
    with open(test_acc_path, "w") as file:
        file.write(str(test_acc))

    return test_acc


def save_boxs_to_file(pr_boxes, pr_scores, gt_boxes, id, iter, args):
    if args.sem_uni:
        model_str = args.model + "-" + args.sgg_model + "-SU"
    else:
        model_str = args.model + "-" + args.sgg_model
    pr_path = "result_sggtest/{}_comp{}_seed{}/{}/prediction/{}_vg{}.txt".format(
        args.dataset, args.complexity, args.seed, model_str, iter, id
    )

    gt_path = "result_sggtest/{}_comp{}_seed{}/{}/ground_truth/{}_vg{}.txt".format(
        args.dataset, args.complexity, args.seed, model_str, iter, id
    )

    # make directories if they do not exist yet
    os.makedirs(
        "result_sggtest/{}_comp{}_seed{}/{}/prediction".format(
            args.dataset, args.complexity, args.seed, model_str
        ),
        exist_ok=True,
    )
    os.makedirs(
        "result_sggtest/{}_comp{}_seed{}/{}/ground_truth".format(
            args.dataset, args.complexity, args.seed, model_str
        ),
        exist_ok=True,
    )
    # save prediction
    text = ""
    # pr_scores = pr_sc
    for i, box in enumerate(pr_boxes):
        try:
            prob = np.round(pr_scores[i].detach().cpu().numpy(), 2)
            text += "target {} {} {} {} {}".format(prob, box[0], box[1], box[2], box[3])
            text += "\n"
        except IndexError:
            continue
    if text[-1] == "\n":
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


def compute_test_accuracy(args, deisam, vg_1, vg_2, data_loader, iter, n_data=400):
    counter = 0
    predictions = []
    labels = []
    right_or_wrong_list = []
    for (
        id,
        data_index,
        image_id,
        image_source,
        image,
        deictic_representation,
        answer,
    ) in data_loader:
        if counter < args.start:
            counter += 1
            continue
        if counter > args.end:
            break
        if counter > n_data:
            break
        try:
            graph_1 = vg_1.load_scene_graph_by_id(image_id)
            graph_2 = vg_2.load_scene_graph_by_id(image_id)
        except KeyError:
            print("Skipped!! ID:{}, IMAGE ID:{}".format(counter, image_id))
            counter += 1
            continue

        print("Deictic representation:")
        print("    " + deictic_representation)
        graphs = [graph_1, graph_2]
        try:
            masks, target_scores, sgg_rules = deisam.forward(
                data_index,
                image_id,
                graphs,
                deictic_representation,
                image_source,
                # llm_rules,
            )
        except openai.error.APIError:
            print("OpenAI API error.. skipping..")
            counter += 1
            continue

        except (openai.InvalidRequestError, openai.error.ServiceUnavailableError):
            print(
                "Got openai.InvalidRequestError or openai.error.ServiceUnavailableError for embeddings in Semantic Unification, skipping.."
            )
            counter += 1
            continue

        if masks == None:
            print(
                "!!!!! No targets founde on image {}. Getting a random mask...".format(
                    counter
                )
            )
            # get a random object's mask
            target_atoms = get_random_masks(deisam)
            masks = deisam.segment_objects_by_sam(image_source, target_atoms, image_id)
            target_scores = [torch.tensor(0.5).to(device)]

        # pr_boxes, gt_boxes = save_box_results(args, masks, answer, id, counter)
        try:
            predicted_boxes = masks_to_boxes(masks.squeeze(1).to(torch.int32))
            answer_boxes = torch.tensor(answer_to_boxes(answer), device=device).to(
                torch.int32
            )
        except RuntimeError:
            answer_boxes = torch.tensor(answer_to_boxes(answer), device=device).to(
                torch.int32
            )
            dummy_box = torch.tensor([0, 0, 1, 1]).to(torch.int32).to(device)
            predicted_boxes = torch.stack([dummy_box for a in answer_boxes])
            print("dummpy predicted boxes are generated...")
        # convert to labeled examples for binary-cross entropy loss

        print("Saving predicted boxes and answers to file ...")
        save_boxs_to_file(
            pr_boxes=predicted_boxes,
            pr_scores=target_scores,
            gt_boxes=answer_boxes,
            id=id,
            iter=counter,
            args=args,
        )
        counter += 1
    return 0, 0

    # compute average
    # acc = np.average(np.array(right_or_wrong_list))
    # return acc, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start",
        help="Start point (data index) for the inference.",
        required=False,
        action="store",
        dest="start",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-e",
        "--end",
        help="End point (data index) for the inference.",
        required=False,
        action="store",
        dest="end",
        type=int,
        default=400,
    )  # 9999)

    parser.add_argument(
        "-ep",
        "--epochs",
        help="Training epochs.",
        required=False,
        action="store",
        dest="epochs",
        type=int,
        default=1,
    )  # 9999)

    parser.add_argument(
        "-sd",
        "--seed",
        help="Random seed.",
        required=False,
        action="store",
        dest="seed",
        type=int,
        default=0,
    )  # 9999)

    parser.add_argument("--lr", type=float, default=1e-2, help="The learning rate.")

    parser.add_argument(
        "-c",
        "-complexity",
        help="The complexity of the DeiVG dataset, i.e. the number of hops in the scene graphs.",
        action="store",
        dest="complexity",
        type=int,
        default=2,
        choices=[1, 2, 3],
    )
    parser.add_argument(
        "-d",
        "--dataset",
        help="Dataset to be used.",
        action="store",
        dest="dataset",
        default="deictic_visual_genome",
        choices=["deictic_visual_genome", "deictic_visual_genome_short"],
    )
    parser.add_argument(
        "-m",
        "--model",
        help="model to be used, DeiSAM or GroundedSAM",
        action="store",
        dest="model",
        default="DeiSAM",
        choices=["DeiSAM", "GroundedSAM"],
    )

    parser.add_argument(
        "-sgg",
        "--sgg-model",
        help="Scene Graph Generation model to be used, None, VETO or [TODO]",
        action="store",
        dest="sgg_model",
        default="VETO",
        choices=["VETO"],
    )

    parser.add_argument(
        "-su",
        "--sem-uni",
        help="Use semantic unifier.",
        action="store_true",
        dest="sem_uni",
    )

    parser.add_argument(
        "-k", "--api-key", help="An OpenAI API key.", action="store", dest="api_key"
    )
    args = parser.parse_args()

    device = torch.device("cuda:0")

    # setup the random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_loader = DeicticVisualGenomeSGGTraining(args, mode="train")
    val_data_loader = DeicticVisualGenomeSGGTraining(args, mode="val")
    test_data_loader = DeicticVisualGenomeSGGTraining(args, mode="test")

    print("Segmenting ...")
    # visual genome utils

    if args.model == "DeiSAM":
        # use ground truth scene graphs
        vg_1 = VisualGenomeUtils()
        vg_2 = PredictedSceneGraphUtils(args.sgg_model)

        deisam = TrainableDeiSAM(
            api_key=args.api_key,
            device=device,
            vg_utils_list=[vg_1, vg_2],
            sem_uni=args.sem_uni,
        )

        results = test_deisam_sgg(
            args=args,
            deisam=deisam,
            data_loader=data_loader,
            val_data_loader=val_data_loader,
            test_data_loader=test_data_loader,
            vg_1=vg_1,
            vg_2=vg_2,
            start_id=args.start,
            end_id=args.end,
            device=device,
            epochs=args.epochs,
        )
    print("Segmentation completed.")
