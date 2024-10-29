import argparse
import os
import random
import numpy as np
import torch
from torchvision.ops import masks_to_boxes
import openai

from data_vg import DeicticVisualGenomeSGGTraining, VisualGenomeUtils, PredictedSceneGraphUtils
from deisam import TrainableDeiSAM
from deisam_utils import get_random_masks, save_segmentation_result_with_alphas
from learning_utils import to_bce_examples
from rtpt import RTPT
from visualization_utils import answer_to_boxes, save_box_to_file

torch.set_num_threads(10)


class DeiSAMTrainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self._set_random_seeds()

        self.data_loader = DeicticVisualGenomeSGGTraining(args, mode="train")
        self.val_data_loader = DeicticVisualGenomeSGGTraining(args, mode="val")
        self.test_data_loader = DeicticVisualGenomeSGGTraining(args, mode="test")

        self.vg_1 = VisualGenomeUtils()
        self.vg_2 = PredictedSceneGraphUtils(args.sgg_model)
        
        self.deisam = TrainableDeiSAM(api_key=args.api_key, device=device,
                                      vg_utils_list=[self.vg_1, self.vg_2],
                                      sem_uni=args.sem_uni)

    def _set_random_seeds(self):
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)

    def train(self):
        rtpt = self._initialize_rtpt()
        optimizer = torch.optim.RMSprop(self.deisam.parameters(), lr=self.args.lr)
        bce_loss = torch.nn.BCELoss()

        os.makedirs(f"learning_logs/comp{self.args.complexity}", exist_ok=True)

        for epoch in range(self.args.epochs):
            for counter, (id, data_index, image_id, image_source, image, 
                          deictic_representation, answer) in enumerate(self.data_loader):
                if self.args.trained:
                    self._load_trained_model()
                    return

                if counter % 25 == 0:
                    self._save_intermediate_model(counter)

                loss = self._process_training_step(counter, data_index, image_id, 
                                                   image_source, deictic_representation, 
                                                   answer, optimizer, bce_loss)
                rtpt.step(subtitle=f"Iter:{counter}")

            self._evaluate_on_test_data(counter)

    def _initialize_rtpt(self):
        rtpt = RTPT(name_initials="", 
                    experiment_name=f"LearnDeiSAM{self.args.complexity}", 
                    max_iterations=100)
        rtpt.start()
        return rtpt

    def _load_trained_model(self):
        save_path = f"models/comp{self.args.complexity}_iter100_seed{self.args.seed}.pth"
        saved_state = torch.load(save_path)
        trained_weights = saved_state["rule_weights"].to(self.device)
        self.deisam.rule_weights = torch.nn.Parameter(trained_weights).to(self.device)

    def _save_intermediate_model(self, counter):
        save_path = f"models/comp{self.args.complexity}_iter{counter}_seed{self.args.seed}.pth"
        torch.save(self.deisam.state_dict(), save_path)
        print(f"Intermediate model saved to {save_path}")

    def _process_training_step(self, counter, data_index, image_id, image_source, 
                               deictic_representation, answer, optimizer, bce_loss):
        print(f"=========== ID {counter}, IMAGE ID {image_id} ===========")
        print("Deictic representation:\n", deictic_representation)

        try:
            graphs = [self.vg_1.load_scene_graph_by_id(image_id),
                      self.vg_2.load_scene_graph_by_id(image_id)]
            masks, target_scores, _ = self.deisam.forward(
                data_index, image_id, graphs, deictic_representation, image_source)

            if masks is None:
                print("No targets segmented.. skipping..")
                return

            predicted_boxes = masks_to_boxes(masks.squeeze(1).float().to(torch.int32))
            answer_boxes = torch.tensor(answer_to_boxes(answer), device=self.device).to(torch.int32)

            box_probs, box_labels = to_bce_examples(predicted_boxes, target_scores, answer_boxes, self.device)
            loss = bce_loss(box_probs, box_labels)
            loss.backward()
            optimizer.step()
            return loss.item()

        except (KeyError, openai.error.APIError, openai.InvalidRequestError, openai.error.ServiceUnavailableError):
            print(f"Skipped or error occurred for ID {counter}, IMAGE ID {image_id}")

    def _evaluate_on_test_data(self, counter):
        segment_testdata(self.args, self.deisam, 
                         self.vg_1, self.vg_2, 
                         self.test_data_loader, 
                         counter, self.device)


def segment_testdata(args, deisam, vg_1, vg_2, data_loader, iter, device, n_data=400):
    for counter, (id, data_index, image_id, image_source, image, deictic_representation, answer) in enumerate(data_loader):
        if counter < args.start or counter > args.end or counter > n_data:
            continue

        print(f"===== TEST ID:{counter}, IMAGE ID:{image_id}")
        print("Deictic representation:\n", deictic_representation)
        
        try:
            graphs = [vg_1.load_scene_graph_by_id(image_id),
                      vg_2.load_scene_graph_by_id(image_id)]
            masks, target_scores, _ = deisam.forward(
                data_index, image_id, graphs, deictic_representation, image_source)

            if masks is None:
                print(f"No targets found on image {counter}. Using random mask.")
                target_atoms = get_random_masks(deisam)
                masks = deisam.segment_objects_by_sam(image_source, target_atoms, image_id)
                target_scores = [torch.tensor(0.5).to(device)]

            predicted_boxes = masks_to_boxes(masks.squeeze(1).to(torch.int32))
            answer_boxes = torch.tensor(answer_to_boxes(answer), device=device).to(torch.int32)

            # TODO: specify the path not to mix up with the solve script results
            save_box_to_file(
                pr_boxes=predicted_boxes,
                pr_scores=target_scores,
                gt_boxes=answer_boxes,
                id=id,
                index=counter,
                iter=iter,
                args=args
            )

            target_scores_cpu = [x.detach().cpu().numpy() for x in target_scores]
            save_segmentation_result_with_alphas(
                args,
                masks,
                target_scores_cpu,
                answer,
                image_source,
                counter,
                image_id,
                data_index,
                deictic_representation,
                iter,
            )
        except openai.error.APIError:
            print("OpenAI API error.. skipping..")

        
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--start", type=int, default=0, help="Start point (data index) for the inference.")
    parser.add_argument("-e", "--end", type=int, default=400, help="End point (data index) for the inference.")
    parser.add_argument("-ep", "--epochs", type=int, default=1, help="Training epochs.")
    parser.add_argument("-sd", "--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--lr", type=float, default=1e-2, help="The learning rate.")
    parser.add_argument("-c", "--complexity", type=int, default=2, choices=[1, 2, 3], help="Complexity level.")
    parser.add_argument("-d", "--dataset", default="deictic_visual_genome", choices=["deictic_visual_genome", "deictic_visual_genome_short"], help="Dataset.")
    parser.add_argument("-m", "--model", default="DeiSAM", choices=["DeiSAM", "GroundedSAM"], help="Model to use.")
    parser.add_argument("-sgg", "--sgg-model", default="", choices=["", "VETO"], help="Scene Graph Generation model.")
    parser.add_argument("-su", "--sem-uni", action="store_true", help="Use semantic unifier.")
    parser.add_argument("-tr", "--trained", action="store_true", help="Use trained model.")
    parser.add_argument("-k", "--api-key", type=str, required=True, help="An OpenAI API key.")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer = DeiSAMTrainer(args, device)

    print("Starting training...")
    trainer.train()
    print("Training completed.")

if __name__ == "__main__":
    main()