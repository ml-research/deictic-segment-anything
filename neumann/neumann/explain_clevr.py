import argparse
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from rtpt import RTPT
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from explanation_utils import *
from logic_utils import get_lang
from neumann_utils import get_data_loader, get_model, get_prob

torch.set_num_threads(10)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Batch size to infer with")
    parser.add_argument("--batch-size-bs", type=int,
                        default=1, help="Batch size in beam search")
    parser.add_argument("--num-objects", type=int, default=6,
                        help="The maximum number of objects in one image")
    parser.add_argument("--dataset", default="delete")  # , choices=["member"])
    parser.add_argument("--dataset-type", default="behind-the-scenes")
    parser.add_argument('--device', default='cpu',
                        help='cuda device, i.e. 0 or cpu')
    parser.add_argument("--no-cuda", action="store_true",
                        help="Run on CPU instead of GPU (not recommended)")
    parser.add_argument("--no-train", action="store_true",
                        help="Perform prediction without training model")
    parser.add_argument("--small-data", action="store_true",
                        help="Use small training data.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of threads for data loader")
    parser.add_argument('--gamma', default=0.01, type=float,
                        help='Smooth parameter in the softor function')
    parser.add_argument("--plot", action="store_true",
                        help="Plot images with captions.")
    parser.add_argument("--t-beam", type=int, default=4,
                        help="Number of rule expantion of clause generation.")
    parser.add_argument("--n-beam", type=int, default=5,
                        help="The size of the beam.")
    parser.add_argument("--n-max", type=int, default=50,
                        help="The maximum number of clauses.")
    parser.add_argument("--program-size", type=int, default=1,
                        help="The size of the logic program.")
    #parser.add_argument("--n-obj", type=int, default=2, help="The number of objects to be focused.")
    parser.add_argument("--epochs", type=int, default=20,
                        help="The number of epochs.")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="The learning rate.")
    parser.add_argument("--n-ratio", type=float, default=1.0,
                        help="The ratio of data to be used.")
    parser.add_argument("--pre-searched", action="store_true",
                        help="Using pre searched clauses.")
    parser.add_argument("--infer-step", type=int, default=6,
                        help="The number of steps of forward reasoning.")
    parser.add_argument("--term-depth", type=int, default=3,
                        help="The number of steps of forward reasoning.")
    parser.add_argument("--question-json-path", default="data/behind-the-scenes/BehindTheScenes_questions.json")
    args = parser.parse_args()
    return args

# def get_nsfr_model(args, lang, clauses, atoms, bk, bk_clauses, device, train=False):


def discretise_NEUMANN(NEUMANN, args, device):
    lark_path = 'src/lark/exp.lark'
    lang_base_path = 'data/lang/'
    lang, clauses_, bk, terms, atoms = get_lang(
        lark_path, lang_base_path, args.dataset_type, args.dataset, args.term_depth)
    # Discretise NEUMANN rules
    clauses = NEUMANN.get_clauses()
    return get_nsfr_model(args, lang, clauses, atoms, bk, bk_clauses, device, train=False)

def predict(NEUMANN, I2F, loader, args, device,  th=None, split='train'):
    predicted_list = []
    target_list = []
    count = 0

    start = time.time()
    for epoch in tqdm(range(args.epochs)):
        for i, sample in enumerate(tqdm(loader), start=0):
            imgs, target_set = map(lambda x: x.to(device), sample)
            # to cuda
            target_set = target_set.float()

            #imgs: torch.Size([1, 1, 3, 128, 128])
            V_0 = I2F(imgs)
            V_T = NEUMANN(V_0)
            #a NEUMANN.print_valuation_batch(V_T)
            predicted = get_prob(V_T, NEUMANN, args)

            # compute explanation for each input image
            for pred in predicted:
                if pred > 0.9:
                    pred.backward(retain_graph=True)
                    atom_grads = NEUMANN.mpm.dummy_zeros.grad.squeeze(-1).unsqueeze(0)
                    attention_maps = I2F.pm.model.slot_attention.attention_maps.squeeze(0)
                    target_attention_maps = get_target_maps(NEUMANN.atoms, atom_grads, attention_maps)
                    #print(atom_grads, torch.max(atom_grads), atom_grads.shape)
                    NEUMANN.print_valuation_batch(atom_grads)

                    

                    imgs_to_plot = to_plot_images_clevr(imgs.squeeze(0).detach().cpu())
                    captions = generate_captions(atom_grads, NEUMANN.atoms, args.num_objects, th=0.33)
                     # + args.dataset + '/' + split + '/', \
                    save_images_with_captions_and_attention_maps(imgs_to_plot, target_attention_maps, captions, folder='explanation/clevr/', \
                                                                 img_id=count, dataset=args.dataset)
                    NEUMANN.mpm.dummy_zeros.grad.detach_()
                    NEUMANN.mpm.dummy_zeros.grad.zero_()
                count += 1
    reasoning_time = time.time() - start
    print('Reasoning Time: ', reasoning_time)
    return 0, 0, 0, reasoning_time


def to_one_label(ys, labels, th=0.7):
    ys_new = []
    for i in range(len(ys)):
        y = ys[i]
        label = labels[i]
        # check in case answers are computed
        num_class = 0
        for p_j in y:
            if p_j > th:
                num_class += 1
        if num_class >= 2:
            # drop the value using label (the label is one-hot)
            drop_index = torch.argmin(label - y)
            y[drop_index] = y.min()
        ys_new.append(y)
    return torch.stack(ys_new)


def main(n):
    seed_everything(n)
    args = get_args()
    assert args.batch_size == 1, "Set batch_size=1."
    #name = 'VILP'
    print('args ', args)
    if args.no_cuda:
        device = torch.device('cpu')
    elif len(args.device.split(',')) > 1:
        # multi gpu
        device = torch.device('cuda')
    else:
        device = torch.device('cuda:' + args.device)

    print('device: ', device)
    name = 'neumann/behind-the-scenes/' + str(n)
    writer = SummaryWriter(f"runs/{name}", purge_step=0)

    # Create RTPT object
    rtpt = RTPT(name_initials='HS', experiment_name=name,
                max_iterations=args.epochs)
    # Start the RTPT tracking
    rtpt.start()


    ## train_pos_loader, val_pos_loader, test_pos_loader = get_vilp_pos_loader(args)
    #####train_pos_loader, val_pos_loader, test_pos_loader = get_data_loader(args)

    # load logical representations
    lark_path = 'src/lark/exp.lark'
    lang_base_path = 'data/lang/'
    lang, clauses, bk, bk_clauses, terms, atoms = get_lang(
        lark_path, lang_base_path, args.dataset_type, args.dataset, args.term_depth, use_learned_clauses=True)

    print("{} Atoms:".format(len(atoms)))

    # get torch data loader
    #question_json_path = 'data/behind-the-scenes/BehindTheScenes_questions_{}.json'.format(args.dataset)
    # test_loader = get_behind_the_scenes_loader(question_json_path, args.batch_size, lang, args.n_data, device)
    train_loader, val_loader, test_loader = get_data_loader(args, device)

    NEUMANN, I2F = get_model(lang=lang, clauses=clauses, atoms=atoms, terms=terms, bk=bk, bk_clauses=bk_clauses,
                          program_size=args.program_size, device=device, dataset=args.dataset, dataset_type=args.dataset_type,
                          num_objects=args.num_objects, infer_step=args.infer_step, train=False, explain=True)#train=not(args.no_train))

    writer.add_scalar("graph/num_atom_nodes", len(NEUMANN.rgm.atom_node_idxs))
    writer.add_scalar("graph/num_conj_nodes", len(NEUMANN.rgm.conj_node_idxs))
    num_nodes = len(NEUMANN.rgm.atom_node_idxs) + len(NEUMANN.rgm.conj_node_idxs)
    writer.add_scalar("graph/num_nodes", num_nodes)

    num_edges = NEUMANN.rgm.edge_index.size(1)
    writer.add_scalar("graph/num_edges", num_edges)

    writer.add_scalar("graph/memory_total", num_nodes + num_edges)

    print("=====================")
    print("NUM NODES: ", num_nodes)
    print("NUM EDGES: ", num_edges)
    print("MEMORY TOTAL: ", num_nodes + num_edges)
    print("=====================")

    params = list(NEUMANN.parameters())
    print('parameters: ', list(params))

    print("Predicting on train data set...")
    times = []
    # train split
    for j in range(n):
        acc_test, rec_test, th_test, time = predict(
            NEUMANN, I2F, test_loader, args, device, th=0.5, split='test')
        times.append(time)
    
    with open('out/inference_time/time_{}_ratio_{}.txt'.format(args.dataset, args.n_ratio), 'w') as f:
        f.write("\n".join(str(item) for item in times))

    print("train acc: ", acc_test, "threashold: ", th_test, "recall: ", rec_test)


def seed_everything(seed: int):
    import os
    import random

    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    main(n=1)

