from huggingface_hub import hf_hub_download
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict
        
def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model   


def segment_dataset_by_SAM(args, data_loader, start_id, end_id):
    # torch.set_num_threads(50)
    # Use this command for evaluate the Grounding DINO model
    # Or you can download the model by yourself
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
    from segment_anything import build_sam, SamPredictor 
    steps = end_id - start_id
    rtpt = RTPT(name_initials='HS', experiment_name='GroSAM', max_iterations=steps)
    rtpt.start()
    
    
    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)
    
    sam_checkpoint = 'sam_vit_h_4b8939.pth'
    sam = build_sam(checkpoint=sam_checkpoint)
    device = torch.device('cuda:0')
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    
    for id, data_index, image_id, image_source, image, deictic_representation, answer  in data_loader:
        if id < start_id:
            continue
        if id > end_id:
            break
        print("========================")
        print("Deictic representation:")
        print("    " + deictic_representation)
        
        boxes, logits, phrases = predict(
            model=groundingdino_model, 
            image=image, 
            caption=deictic_representation, 
            box_threshold=0.3, 
            text_threshold=0.25
        )
        
        sam_predictor.set_image(image_source)
        # box: normalized box xywh -> unnormalized xyxy
        H, W, _ = image_source.shape
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
            save_box_to_file_GroundedSAM(pr_boxes, gt_boxes, id, args.dataset)
            continue
            
        
        # save the result
        pr_boxes = masks_to_boxes(masks.squeeze(1).to(torch.int32))
        gt_boxes = answer_to_boxes(answer)
        save_box_to_file_GroundedSAM(pr_boxes, gt_boxes, id, args.dataset)
            
        logits = [1.0 for box in boxes_xyxy]
        phrases = [""]# prompts
        boxes = torch.stack([torch.tensor(b) for b in boxes])
        # annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        # annotated_frame = annotated_frame[...,::-1] # BGR to RGB
        annotated_frame_with_mask = image_source# annotated_frame
        for i in range(len(masks)):
            # try:
            annotated_frame_with_mask = show_mask(
                    masks[i][0], annotated_frame_with_mask
                )
            # except ValueError:
            #     next
            
        save_segmented_images(id=id,\
            annotated_frame_with_mask=annotated_frame_with_mask, data_index=data_index,\
            deictic_representation=deictic_representation, base_path="plot/{}/GroundedSAM/".format(args.dataset))
        
        rtpt.step(subtitle="ID:{}".format(id))
