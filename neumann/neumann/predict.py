import argparse
import pickle
import random
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from rtpt import RTPT
from sklearn.metrics import accuracy_score, recall_score, roc_curve
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .logic_utils import get_lang
from .neumann_utils import (
    generate_captions,
    get_data_loader,
    get_model,
    get_prob,
    save_images_with_captions,
    to_plot_images_clevr,
    to_plot_images_kandinsky,
)
from .tensor_encoder import TensorEncoder
from .tensor_utils import build_infer_module
from .visualize import plot_proof_history

random.seed(0)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size to infer with"
    )
    parser.add_argument(
        "--num-objects",
        type=int,
        default=3,
        help="The maximum number of objects in one image",
    )
    parser.add_argument("--dataset")  # , choices=["member"])
    parser.add_argument("--dataset_type")  # , choices=["member"])
    parser.add_argument("--rtpt-name", default="")  # , choices=["member"])
    parser.add_argument(
        "--dataset-type",
        choices=["vilp", "clevr-hans", "kandinsky"],
        help="vilp or kandinsky or clevr",
    )
    parser.add_argument("--device", default="cpu", help="cuda device, i.e. 0 or cpu")
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Run on CPU instead of GPU (not recommended)",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Perform prediction without training model",
    )
    parser.add_argument(
        "--small-data", action="store_true", help="Use small training data."
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of threads for data loader"
    )
    parser.add_argument(
        "--gamma",
        default=0.01,
        type=float,
        help="Smooth parameter in the softor function",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Plot images with captions."
    )
    parser.add_argument(
        "--t-beam",
        type=int,
        default=4,
        help="Number of rule expantion of clause generation.",
    )
    parser.add_argument("--n-beam", type=int, default=5, help="The size of the beam.")
    parser.add_argument(
        "--n-max", type=int, default=50, help="The maximum number of clauses."
    )
    parser.add_argument(
        "--program-size",
        "-m",
        type=int,
        default=1,
        help="The size of the logic program.",
    )
    # parser.add_argument("--n-obj", type=int, default=2, help="The number of objects to be focused.")
    parser.add_argument("--epochs", type=int, default=20, help="The number of epochs.")
    parser.add_argument("--lr", type=float, default=1e-2, help="The learning rate.")
    parser.add_argument(
        "--n-data", type=float, default=200, help="The number of data to be used."
    )
    parser.add_argument(
        "--pre-searched", action="store_true", help="Using pre searched clauses."
    )
    parser.add_argument(
        "-T",
        "--infer-step",
        type=int,
        default=10,
        help="The number of steps of forward reasoning.",
    )
    parser.add_argument(
        "--term-depth",
        type=int,
        default=3,
        help="The number of steps of forward reasoning.",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    print("args ", args)
    if args.no_cuda:
        device = torch.device("cpu")
    elif len(args.device.split(",")) > 1:
        # multi gpu
        device = torch.device("cuda")
    else:
        device = torch.device("cuda:" + args.device)

    print("device: ", device)

    # Create RTPT object
    rtpt = RTPT(
        name_initials="",
        experiment_name="NEUMANN_{}".format(args.dataset),
        max_iterations=args.epochs,
    )
    # Start the RTPT tracking
    rtpt.start()

    # Get torch data loader
    # train_loader, val_loader,  test_loader = get_data_loader(args, device)

    # Load logical representations
    lark_path = "src/lark/exp.lark"
    lang_base_path = "data/lang/"
    lang, clauses, bk, bk_clauses, terms, atoms = get_lang(
        lark_path, lang_base_path, args.dataset_type, args.dataset, args.term_depth
    )
    print(terms)
    print("{} Atoms:".format(len(atoms)))
    print(atoms)

    # Load the NEUMANN model
    NEUMANN = get_model(
        lang=lang,
        clauses=clauses,
        atoms=atoms,
        terms=terms,
        bk=bk,
        bk_clauses=bk_clauses,
        program_size=args.program_size,
        device=device,
        dataset=args.dataset,
        dataset_type=args.dataset_type,
        num_objects=args.num_objects,
        term_depth=args.term_depth,
        infer_step=args.infer_step,
        train=not (args.no_train),
    )

    x = torch.zeros((1, len(atoms))).to(device)
    x[:, 0] = 1.0
    ## x[:,1] = 0.8
    print("x: ", x)
    print(np.round(NEUMANN(x).detach().cpu().numpy(), 2))
    print(NEUMANN.rgm.edge_index)
    print("graph: ")
    print(NEUMANN.rgm.networkx_graph)
    NEUMANN.plot_reasoning_graph(name=args.dataset)
    plot_proof_history(
        NEUMANN.mpm.x_atom_list, atoms[1:], args.infer_step, args.dataset, mode="graph"
    )
    # nx.draw(NEUMANN.rgm.networkx_graph)
    # plt.savefig('imgs/{}.png'.format(args.dataset))

    print("==== tensor based reasoner")
    from logic_utils import false

    atoms = [false] + atoms
    rgm = NEUMANN.rgm
    rgm.facts = atoms
    x = torch.zeros((1, len(atoms))).to(device)
    x[:, 1] = 1.0
    for i, atom in enumerate(atoms):
        if atom in bk:
            x[:, i] += 1.0
    ## x[:,2] = 0.8
    IM = build_infer_module(
        clauses,
        atoms,
        lang,
        rgm,
        device,
        m=args.program_size,
        infer_step=args.infer_step,
        train=True,
    )
    IM(x)
    IM.W = NEUMANN.clause_weights
    print(IM.get_weights())
    plot_proof_history(
        IM.V_list, atoms[2:], args.infer_step, args.dataset, mode="tensor"
    )


if __name__ == "__main__":
    main()
