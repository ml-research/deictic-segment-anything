import argparse
import random
import time

import lark
import openai
import rtpt as RTPT
import torch
from data_vg import DeicticVisualGenome, PredictedSceneGraphUtils, VisualGenomeUtils
from deisam import DeiSAM, DeiSAMSGG
from deisam_utils import (
    get_random_masks,
    save_box_results,
    save_llm_response,
    save_segmentation_result,
)
from groundingdino.util.inference import annotate
from rtpt import RTPT
from torchvision.ops import masks_to_boxes
from visualization_utils import answer_to_boxes, save_box_to_file, to_xyxy

torch.set_num_threads(10)


def segment_by_deisam(args, deisam, data_loader, vg, start_id, end_id):
    steps = end_id - start_id
    rtpt = RTPT(
        name_initials="",
        experiment_name="DeiSAM{}".format(args.complexity),
        max_iterations=steps,
    )
    rtpt.start()

    counter = 0
    for (
        id,
        data_index,
        image_id,
        image_source,
        image,
        deictic_representation,
        answer,
    ) in data_loader:
        if counter < start_id:
            counter += 1
            continue
        if counter > end_id:
            break
        print("=========== ID {} ===========".format(counter))
        print("Deictic representation:")
        print("    " + deictic_representation)

        graph = vg.load_scene_graph_by_id(image_id)

        is_failed = False
        try:
            masks, llm_rules, rewritten_rules = deisam.forward(
                data_index, image_id, graph, deictic_representation, image_source
            )
        except KeyError:
            print("Skipped!! ID:{}, IMAGE ID:{}".format(counter, image_id))
            counter += 1
            continue
        except openai.error.RateLimitError:
            print("Got openai.error.RateLimitError, wait for 60s...")
            time.sleep(60)
            masks, llm_rules, rewritten_rules = deisam.forward(
                data_index, image_id, graph, deictic_representation, image_source
            )
        except openai.error.APIError:
            print("Got openai.error.APIError, wait for 20s...")
            time.sleep(20)
            masks, llm_rules, rewritten_rules = deisam.forward(
                data_index, image_id, graph, deictic_representation, image_source
            )

        target_atoms = deisam.target_atoms

        if len(masks) == 0:
            is_failed = True
            print(
                "!!!!! No targets founde on image {}. Getting a random mask...".format(
                    counter
                )
            )
            # get a random object's mask
            target_atoms = get_random_masks(deisam)
            masks = deisam.segment_objects_by_sam(
                image_source, target_atoms, data_index
            )
            # # save failed case
            # with open(
            #     "result/failed_cases_DeiSAM_{}_{}_comp{}.txt".format(
            #         start_id, end_id, args.complexity
            #     ),
            #     "a",
            # ) as f:
            #     f.write(str(counter) + "\n")

        print("Targets: {}".format(str(target_atoms)))
        # save boxes to texts
        print("Saving segmentations as bboxes...")
        pr_boxes, gt_boxes = save_box_results(args, masks, answer, id, counter)

        # save LLM outputs
        print("Saving LLM response to files...")
        save_llm_response(
            args,
            deisam.llm_logic_generator.pred_response,
            deisam.llm_logic_generator.rule_response,
            counter,
            image_id,
            deictic_representation,
        )

        # plot masks to images
        print("Saving the figure with masks...")
        save_segmentation_result(
            args,
            masks,
            answer,
            image_source,
            counter,
            image_id,
            data_index,
            deictic_representation,
            is_failed,
        )
        rtpt.step(subtitle="ID:{}".format(counter))
        counter += 1

    return masks


def segment_by_deisam_with_sgg(args, deisam, data_loader, vg, start_id, end_id):
    steps = end_id - start_id
    rtpt = RTPT(
        name_initials="",
        experiment_name="DeiSAM{}{}".format(args.sgg_model, args.complexity),
        max_iterations=steps,
    )
    rtpt.start()

    counter = 0
    for (
        id,
        data_index,
        image_id,
        image_source,
        image,
        deictic_representation,
        answer,
    ) in data_loader:
        if counter < start_id:
            counter += 1
            continue
        if counter > end_id:
            break
        print("=========== ID:{}, IMAGE ID:{} ===========".format(counter, image_id))
        print("Deictic representation:")
        print("    " + deictic_representation)

        is_failed = False
        try:
            graph = vg.load_scene_graph_by_id(image_id)
            print("Predicted Scene Graph: ")
            masks, llm_rules, rewritten_rules = deisam.forward(
                data_index, image_id, graph, deictic_representation, image_source
            )
        except KeyError:
            print("Skipped!! ID:{}, IMAGE ID:{}".format(counter, image_id))
            counter += 1
            continue
        except openai.error.RateLimitError:
            print("Got openai.error.RateLimitError, wait for 60s...")
            time.sleep(60)
            masks, llm_rules, rewritten_rules = deisam.forward(
                data_index, image_id, graph, deictic_representation, image_source
            )

        target_atoms = deisam.target_atoms

        if len(masks) == 0:
            is_failed = True
            print(
                "!!!!! No targets founde on image {}. Getting a random mask...".format(
                    counter
                )
            )
            # # save failed case
            # with open(
            #     "result/failed_cases_DeiSAM_{}_{}_comp{}.txt".format(
            #         start_id, end_id, args.complexity
            #     ),
            #     "a",
            # ) as f:
            #     f.write(str(counter) + "\n")

        print("Targets: {}".format(str(target_atoms)))
        # save boxes to texts
        print("Saving segmentations as bboxes...")
        pr_boxes, gt_boxes = save_box_results(args, masks, answer, id, counter)

        # plot masks to images
        print("Saving the figure with masks...")
        save_segmentation_result(
            args,
            masks,
            answer,
            image_source,
            counter,
            image_id,
            data_index,
            deictic_representation,
            is_failed,
        )
        rtpt.step(subtitle="ID:{}".format(counter))
        counter += 1

    return masks


def segment_by_groundedsam(args, data_loader, vg, start_id, end_id):
    steps = end_id - start_id
    rtpt = RTPT(
        name_initials="",
        experiment_name="GroundedSAM{}".format(args.complexity),
        max_iterations=steps,
    )
    rtpt.start()

    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
    from segment_anything import SamPredictor, build_sam

    steps = end_id - start_id

    from deisam_utils import load_model_hf

    groundingdino_model = load_model_hf(
        ckpt_repo_id, ckpt_filenmae, ckpt_config_filename
    )

    sam_checkpoint = "sam_vit_h_4b8939.pth"
    sam = build_sam(checkpoint=sam_checkpoint)
    device = torch.device("cuda:0")
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

    counter = 0
    for (
        id,
        data_index,
        image_id,
        image_source,
        image,
        deictic_representation,
        answer,
    ) in data_loader:
        if counter < start_id:
            counter += 1
            continue
        if counter > end_id:
            break
        # if counter == 0:
        #     counter += 1
        #     continue
        print("=========== ID {} ===========".format(counter))
        print("Deictic representation:")
        print("    " + deictic_representation)

        from groundingdino.util.inference import predict

        boxes, logits, phrases = predict(
            model=groundingdino_model,
            image=image,
            caption=deictic_representation,
            box_threshold=0.3,
            text_threshold=0.25,
        )

        sam_predictor.set_image(image_source)
        # box: normalized box xywh -> unnormalized xyxy
        H, W, _ = image_source.shape
        from groundingdino.util import box_ops

        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
        # boxes_xyxy = to_xyxy(boxes) # * torch.Tensor([W, H, W, H])

        transformed_boxes = sam_predictor.transform.apply_boxes_torch(
            boxes_xyxy, image_source.shape[:2]
        ).to(device)

        try:
            masks, _, _ = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
        except RuntimeError:
            # skip because no masks are available
            gt_boxes = answer_to_boxes(answer)
            pr_boxes = torch.tensor([[0, 0, 0, 0]])
            save_box_to_file(pr_boxes, gt_boxes, id, counter, args)
            counter += 1
            continue

        pr_boxes, gt_boxes = save_box_results(args, masks, answer, id, counter)

        # save boxes to texts
        print("Saving segmentations as bboxes...")

        # plot masks to images
        # print("Saving the figure with masks...")
        # save_segmentation_result(
        #     args,
        #     masks,
        #     answer,k
        #     image_source,
        #     counter,
        #     id,
        #     data_index,
        #     deictic_representation,
        #     is_failed,
        # )
        rtpt.step(subtitle="ID:{}".format(counter))
        counter += 1

    return masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
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
        default=10,
    )  # 9999)
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
        default="",
        choices=["", "VETO"],
    )

    parser.add_argument(
        "-k", "--api-key", help="An OpenAI API key.", action="store", dest="api_key"
    )
    args = parser.parse_args()

    device = torch.device("cuda:0")

    data_loader = DeicticVisualGenome(
        path="data/deivg/deictic_vg_v2_comp{}_10k.json".format(args.complexity)
    )

    if args.model == "DeiSAM":
        if args.sgg_model == "":
            # use ground truth scene graphs
            vg = VisualGenomeUtils()
            deisam = DeiSAM(api_key=args.api_key, device=device, vg_utils=vg)
            results = segment_by_deisam(
                args, deisam, data_loader, vg, start_id=args.start, end_id=args.end
            )
        else:
            vg = PredictedSceneGraphUtils(args.sgg_model)
            deisam = DeiSAMSGG(
                api_key=args.api_key,
                device=device,
                vg_utils=vg,
                sgg_model=args.sgg_model,
            )
            results = segment_by_deisam_with_sgg(
                args, deisam, data_loader, vg, start_id=args.start, end_id=args.end
            )
        # deisam
    elif args.model == "GroundedSAM":
        vg = VisualGenomeUtils()
        results = segment_by_groundedsam(
            args, data_loader, vg, start_id=args.start, end_id=args.end
        )
    print("Segmentation completed.")
