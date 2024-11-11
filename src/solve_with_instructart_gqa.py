import argparse
import random
import time
from tqdm import tqdm
import lark
import openai
import rtpt as RTPT
import torch
from data_vg import DeicticVisualGenome, PredictedSceneGraphUtils, VisualGenomeUtils
from sam_utils import to_boxes, to_boxes_with_sgg, to_transformed_boxes
from data_instructart import ArtGQA#, InstructArtAnnotatedGQA
from deisam import DeiSAM, DeiSAMSGG
from deisam_utils import (
    get_random_masks,
    save_box_results,
    # save_box_results_instructart,
    save_llm_response,
    save_segmentation_result,
)
from groundingdino.util.inference import annotate
from rtpt import RTPT
from torchvision.ops import masks_to_boxes
from visualization_utils import answer_to_boxes, save_box_to_file, to_xyxy

torch.set_num_threads(10)


# def segment_by_deisam(args, deisam, data_loader, vg, start_id, end_id):
#     steps = end_id - start_id
#     rtpt = RTPT(
#         name_initials="",
#         experiment_name="DeiSAM{}".format(args.complexity),
#         max_iterations=steps,
#     )
#     rtpt.start()

#     counter = 0
#     for (
#         data_index,
#         image_id,
#         image_source,
#         deictic_representation,
#         answer,
#     ) in data_loader:
#         if counter < start_id:
#             counter += 1
#             continue
#         if counter > end_id:
#             break
#         print("=========== ID {} ===========".format(counter))
#         print("Deictic representation:")
#         print("    " + deictic_representation)

#         graph = vg.load_scene_graph_by_id(image_id)

#         is_failed = False
#         try:
#             masks, llm_rules, rewritten_rules = deisam.forward(
#                 data_index, image_id, graph, deictic_representation, image_source
#             )
#         except KeyError:
#             print("Skipped!! ID:{}, IMAGE ID:{}".format(counter, image_id))
#             counter += 1
#             continue
#         except openai.error.RateLimitError:
#             print("Got openai.error.RateLimitError, wait for 60s...")
#             time.sleep(60)
#             masks, llm_rules, rewritten_rules = deisam.forward(
#                 data_index, image_id, graph, deictic_representation, image_source
#             )
#         except openai.error.APIError:
#             print("Got openai.error.APIError, wait for 20s...")
#             time.sleep(20)
#             masks, llm_rules, rewritten_rules = deisam.forward(
#                 data_index, image_id, graph, deictic_representation, image_source
#             )
#         except IndexError:
#             print("Skipped!! ID:{}, IMAGE ID:{}".format(counter, image_id))
#             counter += 1
#             continue
            

#         target_atoms = deisam.target_atoms

#         if len(masks) == 0:
#             is_failed = True
#             print(
#                 "!!!!! No targets founde on image {}. Getting a random mask...".format(
#                     counter
#                 )
#             )
#             # get a random object's mask
#             try:
#                 target_atoms = get_random_masks(deisam)
#                 masks = deisam.segment_objects_by_sam(
#                     image_source, target_atoms, data_index
#                 )
#             except IndexError:
#                 print("Skipped!! ID:{}, IMAGE ID:{}".format(counter, image_id))
#                 counter += 1
#                 continue

#         print("Targets: {}".format(str(target_atoms)))
#         # save boxes to texts
#         print("Saving segmentations as bboxes...")
#         pr_boxes, gt_boxes = save_box_results_instructart(args, masks, answer, id, counter)

#         # save LLM outputs
#         # print("Saving LLM response to files...")
#         # save_llm_response(
#         #     args,
#         #     deisam.llm_logic_generator.pred_response,
#         #     deisam.llm_logic_generator.rule_response,
#         #     counter,
#         #     image_id,
#         #     deictic_representation,
#         # )

#         # plot masks to images
#         print("Saving the figure with masks...")
#         save_segmentation_result(
#             args,
#             masks,
#             answer,
#             image_source,
#             counter,
#             image_id,
#             data_index,
#             deictic_representation,
#             is_failed,
#         )
#         rtpt.step(subtitle="ID:{}".format(counter))
#         counter += 1

#     return masks


def segment_by_art(args, deisam, data_loader, start_id, end_id, device):
    steps = end_id - start_id
    rtpt = RTPT(
        name_initials="HS",
        experiment_name="DeiSAM{}".format(args.complexity),
        max_iterations=steps,
    )
    rtpt.start()

    counter = 0
    for (image_id, image_source, image, deictic_representation, subject_box, answer, target_label) in tqdm(data_loader):
        # if counter % 2000 == 0:
        #     print("Counter: ", counter)
        # if not int(image_id) in [2349401, 2400651, 3381]:
        #     counter += 1
        #     continue
        # answer.to(device)
        if counter < start_id:
            counter += 1
            continue
        if counter > end_id:
            break
        print("=========== ID {} ===========".format(counter))
        print("Deictic representation:")
        print("    " + deictic_representation + "   Answer: " + target_label)
        
        # box = [(answer[0] + answer[2])/2, (answer[1] + answer[3])/2, answer[2] - answer[0], answer[3] - answer[1]]
        # box = [answer]
        
        try:
            boxes = torch.tensor([answer]).to(deisam.device)
        except RuntimeError:
            # No answer is matched
            # Regard that ART used wrong object as the answer
            print("No answer is matched. Using the subject box as the answer.")
            print("Skipping...")
            counter += 1
            continue
            # boxes = torch.tensor([[0, 0, 0, 0]]).to(deisam.device)
        
        # set image source
        deisam.sam_predictor.set_image(image_source)
        H, W, _ = image_source.shape
        from groundingdino.util import box_ops

        # boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
        # boxes_xyxy = to_xyxy(boxes) # * torch.Tensor([W, H, W, H])

        transformed_boxes = deisam.sam_predictor.transform.apply_boxes_torch(
            boxes, image_source.shape[:2]
        ).to(deisam.device)
        
        # transformed_boxes = boxes

        
        masks, _, _ = deisam.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )


        # save boxes to texts
        print("Saving segmentations as bboxes...")
        masks = masks.squeeze(dim=1).to(torch.int32)
        assert torch.max(masks) == 1, "Masks contain no active pixels"
        
        pr_boxes = masks_to_boxes(masks)
        gt_boxes = [answer] # answer is alraedy in xyxy format
        save_box_to_file(pr_boxes, gt_boxes, image_id, counter, args)

        is_failed = False
        # plot masks to images
        print("Saving the figure with masks...")
        save_segmentation_result(
            args,
            masks,
            image_source,
            counter,
            image_id,
            deictic_representation + "_" + target_label,
            is_failed,
        )
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
        default=1,
        choices=[1],
    )
    parser.add_argument(
        "-d",
        "--dataset",
        help="Dataset to be used.",
        action="store",
        dest="dataset",
        default="gqa",
        # choices=["deictic_visual_genome", "deictic_visual_genome_short"],
    )
    parser.add_argument(
        "-m",
        "--model",
        help="model to be used, DeiSAM or GroundedSAM",
        action="store",
        dest="model",
        default="ART",
        choices=["DeiSAM", "GroundedSAM", "ART"],
    )

    parser.add_argument(
        "-sgg",
        "--sgg-model",
        help="Scene Graph Generation model to be used, None, VETO or [TODO]",
        action="store",
        dest="sgg_model",
        default="",
        choices=["", "VETO", "ART"],
    )

    parser.add_argument(
        "-k", "--api-key", help="An OpenAI API key.", action="store", dest="api_key"
    )
    args = parser.parse_args()

    device = torch.device("cuda:0")
    # device = torch.device("cpu")


    # if args.model == "DeiSAM":
        
    #     data_loader = InstructArtGQA(
    #     # path="data/deictic_vg_v2_comp{}_10k.json".format(args.complexity)
    #     )
    #     if args.sgg_model == "":
    #         # use ground truth scene graphs
    #         vg = VisualGenomeUtils()
    #         deisam = DeiSAM(api_key=args.api_key, device=device, vg_utils=vg, sem_uni=False)
    #         results = segment_by_deisam(
    #             args, deisam, data_loader, vg, start_id=args.start, end_id=args.end
    #         )
    #     else:
    #         vg = PredictedSceneGraphUtils(args.sgg_model)
    #         deisam = DeiSAMSGG(
    #             api_key=args.api_key,
    #             device=device,
    #             vg_utils=vg,
    #             sgg_model=args.sgg_model,
    #         )
    #         # results = segment_by_deisam_with_sgg(
    #         #     args, deisam, data_loader, vg, start_id=args.start, end_id=args.end
    #         # )
    #     # deisam
    # elif args.model == "InstructArt":
    data_loader = ArtGQA()
    # vg = VisualGenomeUtils()
    deisam = DeiSAM(api_key=args.api_key, device=device, vg_utils=None)
    results = segment_by_art(
        args, deisam, data_loader, start_id=args.start, end_id=args.end, device=device
    )
    print("Segmentation completed.")
