import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .facts_converter import FactsConverter, FactsConverterWithQuery
from .img2facts import Img2Facts, Img2FactsWithQuery
from .logic_utils import add_true_atom
from .message_passing import MessagePassingModule
from .neumann import NEUMANN
from .percept import (
    SlotAttentionLessColorsPerceptionModule,
    SlotAttentionPerceptionModule,
    YOLOPerceptionModule,
)
from .reasoning_graph import ReasoningGraphModule
from .soft_logic import SoftLogic
from .valuation import (
    SlotAttentionValuationModule,
    SlotAttentionWithQueryValuationModule,
    YOLOValuationModule,
)


def get_neumann_model(
    clauses, bk_clauses, bk, atoms, terms, lang, term_depth, infer_step, device
):
    RGM = load_reasoning_graph(
        clauses,
        bk_clauses,
        atoms,
        terms,
        lang,
        term_depth,
        device,
    )
    # node feature module
    # build Reasoning GNN
    soft_logic = SoftLogic()
    # (in_channels=args.node_dim, out_channels=len(atoms)
    MPM = MessagePassingModule(soft_logic, device, T=infer_step)
    NEUM = NEUMANN(
        atoms=atoms,
        clauses=clauses,
        message_passing_module=MPM,
        reasoning_graph_module=RGM,
        bk=bk,
        bk_clauses=bk_clauses,
        device=device,
        program_size=1,
        train=False,
    )
    return NEUM


def get_trainable_neumann_model(
    clauses, bk_clauses, bk, atoms, terms, lang, term_depth, infer_step, device
):
    RGM = load_reasoning_graph(
        clauses,
        bk_clauses,
        atoms,
        terms,
        lang,
        term_depth,
        device,
    )
    # node feature module
    # build Reasoning GNN
    soft_logic = SoftLogic()
    # (in_channels=args.node_dim, out_channels=len(atoms)
    MPM = MessagePassingModule(soft_logic, device, T=infer_step)
    NEUM = NEUMANN(
        atoms=atoms,
        clauses=clauses,
        message_passing_module=MPM,
        reasoning_graph_module=RGM,
        bk=bk,
        bk_clauses=bk_clauses,
        device=device,
        program_size=1,
        train=True,
    )
    return NEUM


def load_reasoning_graph(clauses, bk_clauses, atoms, terms, lang, term_depth, device):
    RGM = ReasoningGraphModule(
        clauses=clauses + bk_clauses,
        facts=atoms,
        terms=terms,
        lang=lang,
        max_term_depth=term_depth,
        device=device,
    )
    return RGM


def update_by_clauses(neumann, clauses, softmax_temp=1.0):
    """Generate new neumann instance by generating and adding new clauses using clause scores."""
    ### Do we need old clauses?? too general clauses should be excluded

    RGM = ReasoningGraphModule(
        clauses=clauses + neumann.bk_clauses,
        facts=neumann.atoms,
        terms=neumann.rgm.terms,
        lang=neumann.rgm.lang,
        max_term_depth=neumann.rgm.max_term_depth,
        device=neumann.device,
        clause_casche=neumann.rgm.clause_casche,
        grounding_casche=neumann.rgm.grounding_casche,
    )
    NEUM = NEUMANN(
        atoms=neumann.atoms,
        clauses=clauses,
        message_passing_module=neumann.mpm,
        reasoning_graph_module=RGM,
        bk=neumann.bk,
        bk_clauses=neumann.bk_clauses,
        device=neumann.device,
        program_size=neumann.program_size,
        train=neumann.train,
        softmax_tmp=softmax_temp,
    )
    return NEUM


def update_by_refinement(
    neumann, clause_scores, clause_generator, softmax_temp=1.0, replace=False
):
    """Generate new neumann instance by generating and adding new clauses using clause scores."""
    # printing with scores
    idxs = np.argsort(-clause_scores.cpu().numpy())
    print(clause_scores)
    print("==== CLAUSE SCORES ===")
    for n, i in enumerate(idxs):
        # if n < 20:
        print(np.round(clause_scores[i].cpu().numpy(), 3), neumann.clauses[i])
    generated_clauses = clause_generator.generate(neumann.clauses, clause_scores)
    ### clause_generator.print_tree()
    pruned_old_clauses = [
        c
        for c in neumann.clauses
        if not add_true_atom(c) in clause_generator.refinement_history
    ]
    ### Do we need old clauses?? too general clauses should be excluded
    # new_clauses = sorted(list(set(neumann.clauses + new_gen_clauses)))
    if replace:
        new_clauses = sorted(list(set(generated_clauses)))
    else:
        new_clauses = sorted(list(set(generated_clauses + pruned_old_clauses)))

    RGM = ReasoningGraphModule(
        clauses=new_clauses + neumann.bk_clauses,
        facts=neumann.atoms,
        terms=neumann.rgm.terms,
        lang=neumann.rgm.lang,
        max_term_depth=neumann.rgm.max_term_depth,
        device=neumann.device,
    )  # , clause_casche=neumann.rgm.clause_casche, grounding_casche=neumann.rgm.grounding_casche)
    NEUM = NEUMANN(
        atoms=neumann.atoms,
        clauses=new_clauses,
        message_passing_module=neumann.mpm,
        reasoning_graph_module=RGM,
        bk=neumann.bk,
        bk_clauses=neumann.bk_clauses,
        device=neumann.device,
        program_size=neumann.program_size,
        train=neumann.train,
        softmax_tmp=softmax_temp,
    )
    return NEUM, generated_clauses


def get_model(
    lang,
    clauses,
    atoms,
    terms,
    bk,
    bk_clauses,
    program_size,
    device,
    dataset,
    dataset_type,
    num_objects,
    term_depth=6,
    infer_step=10,
    train=False,
    explain=False,
):
    if dataset_type in ["synthetic"]:
        RGM = load_reasoning_graph(
            clauses,
            bk_clauses,
            atoms,
            terms,
            lang,
            term_depth,
            device,
            dataset,
            dataset_type,
        )
        # node feature module
        # build Reasoning GNN
        soft_logic = SoftLogic()
        # (in_channels=args.node_dim, out_channels=len(atoms)
        MPM = MessagePassingModule(soft_logic, device, T=infer_step)
        NEUM = NEUMANN(
            atoms=atoms,
            clauses=clauses,
            message_passing_module=MPM,
            reasoning_graph_module=RGM,
            bk=bk,
            bk_clauses=bk_clauses,
            device=device,
            program_size=program_size,
            train=train,
        )
        return NEUM
    elif dataset_type in ["vilp", "clevr-hans"]:
        print("Loading SlotAttention Perception Module...")
        PM = SlotAttentionPerceptionModule(e=num_objects, d=19, device=device).to(
            device
        )
        VM = SlotAttentionValuationModule(lang=lang, device=device)
        FC = FactsConverter(
            lang=lang,
            atoms=atoms,
            bk=bk,
            perception_module=PM,
            valuation_module=VM,
            device=device,
        )
        I2F = Img2Facts(
            perception_module=PM, facts_converter=FC, atoms=atoms, bk=bk, device=device
        )
    elif dataset_type in ["behind-the-scenes"]:
        print("Loading SlotAttention Perception Module...")
        PM = SlotAttentionLessColorsPerceptionModule(
            e=num_objects, d=19, device=device
        ).to(device)
        VM = SlotAttentionWithQueryValuationModule(lang=lang, device=device)
        FC = FactsConverterWithQuery(
            lang=lang,
            atoms=atoms,
            bk=bk,
            perception_module=PM,
            valuation_module=VM,
            device=device,
        )
        I2F = Img2FactsWithQuery(
            perception_module=PM, facts_converter=FC, atoms=atoms, bk=bk, device=device
        )
    else:
        print("Loading YOLO Perception Module...")
        PM = YOLOPerceptionModule(e=num_objects, d=11, device=device)
        VM = YOLOValuationModule(lang=lang, device=device, dataset=dataset)
        FC = FactsConverter(
            lang=lang,
            atoms=atoms,
            bk=bk,
            perception_module=PM,
            valuation_module=VM,
            device=device,
        )
        I2F = Img2Facts(
            perception_module=PM, facts_converter=FC, atoms=atoms, bk=bk, device=device
        )
    # build reasoning graph
    # RGM = ReasoningGraphModule(clauses=clauses+bk_clauses, facts=atoms, terms=terms, lang=lang, device=device)
    RGM = load_reasoning_graph(
        clauses,
        bk_clauses,
        atoms,
        terms,
        lang,
        term_depth,
        device,
        dataset,
        dataset_type,
    )
    # node feature module
    # build Reasoning GNN
    soft_logic = SoftLogic()
    # (in_channels=args.node_dim, out_channels=len(atoms)
    MPM = MessagePassingModule(soft_logic, device, T=infer_step)
    NEUM = NEUMANN(
        atoms=atoms,
        clauses=clauses,
        message_passing_module=MPM,
        reasoning_graph_module=RGM,
        bk=bk,
        bk_clauses=bk_clauses,
        device=device,
        program_size=program_size,
        train=train,
        explain=explain,
    )
    return NEUM, I2F


def get_clause_evaluator(lang, clauses, atoms, terms, bk, bk_clauses, device):
    device = torch.device("cpu")
    PM = SlotAttentionPerceptionModule(e=5, d=19, device=device)
    VM = SlotAttentionValuationModule(lang=lang, device=device)
    FC = FactsConverter(
        lang=lang, perception_module=PM, valuation_module=VM, device=device
    )
    I2F = Img2Facts(
        perception_module=PM, facts_converter=FC, atoms=atoms, bk=bk, device=device
    )
    # build reasoning graph
    RGM = ReasoningGraphModule(
        clauses=clauses + bk_clauses, facts=atoms, terms=terms, lang=lang, device=device
    )
    # node feature module
    # build Reasoning GNN
    soft_logic = SoftLogic()
    # (in_channels=args.node_dim, out_channels=len(atoms)
    MPM = MessagePassingModule(soft_logic, device, T=5)
    CE = ClauseEvaluator(
        lang=lang,
        clauses=clauses,
        atoms=atoms,
        terms=terms,
        message_passing_module=MPM,
        bk=bk,
        bk_clauses=bk_clauses,
        device=device,
    )
    return CE


def update_clauses(gnnr, clauses, bk_clauses, device):
    CIM = build_clause_infer_module(
        clauses, bk_clauses, gnnr.atoms, gnnr.fc.lang, m=len(clauses), device=device
    )
    new_gnnr = GNNReasoner(
        perception_module=gnnr.pm,
        facts_converter=gnnr.fc,
        infer_module=gnnr.im,
        clause_infer_module=CIM,
        atoms=gnnr.atoms,
        bk=nsfr.bk,
        clauses=clauses,
    )
    GNNR = GNNReasoner(
        atoms=atoms,
        clauses=clauses,
        message_passing_module=MPM,
        reasoning_graph_module=RGM,
        device=device,
    )
    new_gnnr._summary()
    del GNNR
    return new_gnnr


def get_prob(v_T, Reasoner, args):
    if args.dataset_type == "vilp" and args.dataset in ["member", "reverse", "sort"]:
        return Reasoner.predict_by_atom(v_T, "pos(img1,img2)")
    elif args.dataset_type == "vilp" and args.dataset in ["delete", "append"]:
        return Reasoner.predict_by_atom(v_T, "pos(img1,img2,img3)")
    elif args.dataset_type in ["kandinsky", "clevr-hans"]:
        return Reasoner.predict_by_atom(v_T, "kp(img)")
    elif args.dataset_type in ["behind-the-scenes"]:
        return torch.cat(
            [
                Reasoner.predict_by_atom(v_T, "answer(cyan)").unsqueeze(-1),
                Reasoner.predict_by_atom(v_T, "answer(gray)").unsqueeze(-1),
                Reasoner.predict_by_atom(v_T, "answer(red)").unsqueeze(-1),
                Reasoner.predict_by_atom(v_T, "answer(yellow)").unsqueeze(-1),
            ],
            dim=1,
        )
    else:
        assert 0, "Invalid dataset for get_prob: {}".format(args.dataset_type)


def get_data_loader(args, device, pos_ratio=1.0, neg_ratio=1.0):
    if args.dataset_type == "kandinsky":
        return get_kandinsky_loader(args, pos_ratio, neg_ratio)
    elif args.dataset_type == "clevr-hans":
        return get_clevr_loader(args, pos_ratio, neg_ratio)
    elif args.dataset_type == "vilp":
        return get_vilp_loader(args, pos_ratio, neg_ratio)
    elif args.dataset_type == "behind-the-scenes":
        return get_behind_the_scenes_loader(args, device)
    else:
        assert 0, "Invalid dataset type: " + args.dataset_type


def get_kandinsky_loader(args, pos_ratio, neg_ratio):
    dataset_train = data_kandinsky.KANDINSKY(
        args.dataset, "train", pos_ratio, neg_ratio
    )
    dataset_val = data_kandinsky.KANDINSKY(
        args.dataset, "val", pos_ratio=pos_ratio, neg_ratio=neg_ratio
    )
    dataset_test = data_kandinsky.KANDINSKY(
        args.dataset, "test", pos_ratio=1.0, neg_ratio=1.0
    )

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    return train_loader, val_loader, test_loader


def get_behind_the_scenes_loader(question_json_path, batch_size, lang, n_data, device):
    dataset_test = data_behind_the_scenes.BehindTheScenes(
        question_json_path, lang, n_data, device
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test, shuffle=True, batch_size=batch_size, num_workers=0
    )
    return test_loader


def get_clevr_loader(args, pos_ratio, neg_ratio):
    dataset_train = data_clevr.CLEVRHans(args.dataset, "train", pos_ratio, neg_ratio)
    dataset_val = data_clevr.CLEVRHans(
        args.dataset, "val", pos_ratio=1.0, neg_ratio=1.0
    )
    dataset_test = data_clevr.CLEVRHans(
        args.dataset, "test", pos_ratio=1.0, neg_ratio=1.0
    )

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    return train_loader, val_loader, test_loader


def get_vilp_loader(args, pos_ratio, neg_ratio):
    dataset_train = data_vilp.VisualILP(
        args.dataset, "train", pos_ratio=pos_ratio, neg_ratio=neg_ratio
    )
    dataset_val = data_vilp.VisualILP(args.dataset, "val")
    dataset_test = data_vilp.VisualILP(args.dataset, "test")

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    return train_loader, val_loader, test_loader


def get_vilp_positive_loader(args, pos_ratio, neg_ratio):
    dataset_train = data_vilp.VisualILP_POSITIVE(
        args.dataset, "train", pos_ratio=pos_ratio, neg_ratio=neg_ratio
    )
    dataset_val = data_vilp.VisualILP_POSITIVE(args.dataset, "val")
    dataset_test = data_vilp.VisualILP_POSITIVE(args.dataset, "test")

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    return train_loader, val_loader, test_loader


def save_images_with_captions(imgs, captions, folder, img_id_start, dataset):
    if not os.path.exists(folder):
        os.makedirs(folder)

    if dataset == "online-pair":
        figsize = (15, 15)
    elif dataset == "red-triangle":
        figsize = (10, 8)
    else:
        figsize = (15, 10)
    # imgs should be denormalized.
    img_id = img_id_start
    for i, img in enumerate(imgs):
        plt.figure(figsize=figsize, dpi=80)
        plt.imshow(img)
        plt.xlabel(captions[i], fontsize=14)
        plt.tight_layout()
        plt.savefig(folder + str(img_id) + ".png")
        img_id += 1
        plt.close()


def to_plot_images_clevr(imgs):
    return [
        img.permute(1, 2, 0).cpu().detach().numpy() for img in denormalize_clevr(imgs)
    ]


def to_plot_images_kandinsky(imgs):
    return [
        img.permute(1, 2, 0).cpu().detach().numpy()
        for img in denormalize_kandinsky(imgs)
    ]


def denormalize_clevr(imgs):
    """denormalize clevr images"""
    # normalizing: image = (image - 0.5) * 2.0  # Rescale to [-1, 1].
    return (0.5 * imgs) + 0.5


def denormalize_kandinsky(imgs):
    """denormalize kandinsky images"""
    return imgs


def generate_captions(V, atoms, e, th):
    captions = []
    for v in V:
        # for each data in the batch
        captions.append(valuation_to_string(v, atoms, e, th))
    return captions


def valuation_to_attr_string(v, atoms, e, th=0.5):
    """Generate string explanations of the scene."""

    st = ""
    for i in range(e):
        st_i = ""
        for j, atom in enumerate(atoms):
            # print(atom, [str(term) for term in atom.terms])
            if (
                "obj" + str(i) in [str(term) for term in atom.terms]
                and atom.pred.name in attrs
            ):
                if v[j] > th and not (
                    atom.pred.name
                    in attrs
                    + ["in", ".", "delete", "member", "not_member", "right_most"]
                ):
                    prob = np.round(v[j].detach().cpu().numpy(), 2)
                    st_i += str(prob) + ":" + str(atom) + ","
        if st_i != "":
            st_i = st_i[:-1]
            st += st_i + "\n"
    return st


def valuation_to_rel_string(v, atoms, th=0.5):
    l = 100
    st = ""
    n = 0
    for j, atom in enumerate(atoms):
        if v[j] > th and not (
            atom.pred.name
            in attrs + ["in", ".", "delete", "member", "not_member", "right_most"]
        ):
            prob = np.round(v[j].detach().cpu().numpy(), 2)
            st += str(prob) + ":" + str(atom) + ","
            n += len(str(prob) + ":" + str(atom) + ",")
        if n > l:
            st += "\n"
            n = 0
    return st[:-1] + "\n"


def valuation_to_string(v, atoms, e, th=0.5):
    return valuation_to_attr_string(v, atoms, e, th) + valuation_to_rel_string(
        v, atoms, th
    )


def valuations_to_string(V, atoms, e, th=0.5):
    """Generate string explanation of the scenes."""
    st = ""
    for i in range(V.size(0)):
        st += "image " + str(i) + "\n"
        # for each data in the batch
        st += valuation_to_string(V[i], atoms, e, th)
    return st
