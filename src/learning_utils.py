from tkinter import TRUE

import torch
from torchvision.ops import box_iou

from neumann.fol.language import DataType, Language
from neumann.fol.logic import Atom, Clause, NeuralPredicate, Predicate, Var


def translate_rules_to_sgg_format(rules_list, num_sgg_models=2):
    """Translate FOL rules for the learning setup., e.g.
    Given:
        target(X):-cond1(X),cond2(X).
        cond1(X):-has(X,Y),type(Y,hair).
        cond2(X):-on(X,Y),on(Y,surfboard).

    Output:
        target_sg1(X):-cond1_sg1(X),cond2_sg1(X).
        cond1_sg1(X):-has_sg1(X,Y),type_sg1(Y,hair).
        cond2_sg1(X):-on_sg1(X,Y),on_sg1(Y,surfboard).

        target_sg2(X):-cond1_sg2(X),cond2_sg2_(X).
        cond1_sg2(X):-has_sg2(X,Y),type_sg2(Y,hair).
        cond2_sg2(X):-on_sg2(X,Y),on_sg2(Y,surfboard).

        target(X):-target_sg1(X).
        target(X):-target_sg2(X).

    Args:
        rules (_type_): _description_
        lang (_type_): _description_
    """
    translated_rules = []
    # target_rules = []
    # new_lang = Language(preds=[], funcs=[], consts=lang.consts.copy())
    for i, rules in enumerate(rules_list):
        # for i-th sgg
        for rule in rules:
            # translate cond rules
            new_rule_atoms = []
            for atom in [rule.head] + rule.body:
                # if atom.pred.name != "target":
                new_pred = NeuralPredicate(
                    atom.pred.name + "_sgg{}".format(i),
                    atom.pred.arity,
                    atom.pred.dtypes,
                )
                new_rule_atoms.append(Atom(new_pred, atom.terms))
            # generate new translated rule
            new_rule = Clause(new_rule_atoms[0], new_rule_atoms[1:])
            if new_rule not in translated_rules:
                translated_rules.append(new_rule)
    return translated_rules  # , new_lang


def translate_atoms_to_sgg_format(atoms_list, langs):
    """Translate lists of atoms, List[List[Atoms]], to indivisual representations for each scene graph generator, e.g.
    Given:
            [[cond1(obj1), on(obj1,obj2)] , [cond1(obj1), on(obj1,obj2)]]
    Return:
            [[cond1_sg1(obj1), on_sg1(obj1,obj2)] , [cond1_sg2(obj1), on_sg2(obj1,obj2)]]

    Args:
        atoms_list (_type_): _description_
    """
    # num_sgg_model = len(atoms_list)
    translated_atoms = []
    new_lang = Language(preds=[], funcs=[], consts=[])
    # init merged language
    for lang in langs:
        for p in lang.preds.copy():
            if p not in new_lang.preds:
                new_lang.preds.append(p)
        for c in lang.consts.copy():
            if c not in new_lang.consts:
                new_lang.consts.append(c)
        # new_lang.preds.append(lang.preds.copy())

    # new_lang = Language(
    #     preds=langs[i].preds.copy(), funcs=[], consts=langs.consts.copy()
    # )

    for i, atoms in enumerate(atoms_list):
        # i-th sgg
        for atom in atoms:
            # pred name
            if atom.pred.name == ".":
                # the special true atom
                if atom not in translated_atoms:
                    translated_atoms.append(atom)
                continue
            # for normal atoms
            pred_name = atom.pred.name + "_sgg{}".format(i)
            new_pred = Predicate(pred_name, atom.pred.arity, atom.pred.dtypes)
            new_atom = Atom(new_pred, atom.terms)
            if new_atom not in translated_atoms:
                translated_atoms.append(new_atom)
            # update language with new predicate
            if new_pred not in new_lang.preds:
                new_lang.preds.append(new_pred)
    return translated_atoms, new_lang


def get_target_selection_rules(lang, num_sgg_models=2):
    """Generate selection rules over scene graph generators, e.g.
            target(X):-target_sg1(X).
            target(X):-target_sg2(X).

    Args:
        num_sgg_models (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    new_lang = Language(preds=lang.preds.copy(), funcs=[], consts=lang.consts.copy())
    dtype = DataType("object")
    target_pred = Predicate("target", 1, [dtype])
    new_lang.preds.append(target_pred)
    target_atom = Atom(target_pred, [Var("X")])
    selection_rules = []
    for i in range(num_sgg_models):
        target_pred_sg_i = Predicate("target_sgg{}".format(i), 1, [dtype])
        new_lang.preds.append(target_pred_sg_i)
        target_atom_sg_i = Atom(target_pred_sg_i, [Var("X")])
        target_rule_sg_i = Clause(target_atom, [target_atom_sg_i])
        selection_rules.append(target_rule_sg_i)
    return selection_rules, new_lang
    # add target preds

    # for i in range(num_sgg_models):
    #     new_cond_pred = NeuralPredicate(
    #         rule.head.pred.name + "_sg{}".format(i),
    #         rule.head.pred.arity,
    #         rule.head.pred.dtypes,
    #     )
    #     for body_atom in rule.body:
    #         # if not body_atom.pred.name in ["type"]:
    #         # has_sg1
    #         new_body_pred = NeuralPredicate(
    #             body_atom.pred.name + "_sg{}".format(i),
    #             body_atom.pred.arity,
    #             body_atom.pred.dtypes,
    #         )


def merge_langs(langs):
    preds = []
    consts = []
    for lang in langs:
        preds.extend(lang.preds.copy())
        consts.extend(lang.consts.copy())
    new_lang = Language(preds=preds, funcs=[], consts=consts)
    return new_lang


def merge_atoms_list(atoms_list):
    merged_atoms = []
    for atoms in atoms_list:
        for atom in atoms:
            if not atom in merged_atoms:
                merged_atoms.append(atom)
    return merged_atoms


def __translate_rules_to_sgg_format(rules, num_sgg_models=2):
    """Translate FOL rules for the learning setup., e.g.
    Given:
        target(X):-cond1(X),cond2(X).
        cond1(X):-has(X,Y),type(Y,hair).
        cond2(X):-on(X,Y),on(Y,surfboard).

    Output:
        target_sg1(X):-cond1_sg1(X),cond2_sg1(X).
        cond1_sg1(X):-has_sg1(X,Y),type_sg1(Y,hair).
        cond2_sg1(X):-on_sg1(X,Y),on_sg1(Y,surfboard).

        target_sg2(X):-cond1_sg2(X),cond2_sg2_(X).
        cond1_sg2(X):-has_sg2(X,Y),type_sg2(Y,hair).
        cond2_sg2(X):-on_sg2(X,Y),on_sg2(Y,surfboard).

        target(X):-target_sg1(X).
        target(X):-target_sg2(X).

    Args:
        rules (_type_): _description_
        lang (_type_): _description_
    """
    translated_rules = []
    # target_rules = []
    # new_lang = Language(preds=[], funcs=[], consts=lang.consts.copy())
    for rule in rules:
        # atoms = [rule.head] + rule.body
        # if rule.head.pred.name != "target" and "cond" in rule.head.pred.name:
        # translate cond rules
        new_rule_atoms = []
        for atom in rule.body:
            # if atom.pred.name != "target":
            new_pred = NeuralPredicate(
                atom.pred.name + "_sg{}".format(i),
                atom.pred.arity,
                atom.pred.dtypes,
            )
            # extend the language with new predicates
            # new_lang.preds.append(new_pred)
            # delete new predicate? TODO
            # save new translated rule
            new_rule_atoms.append(Atom(new_pred, atom.terms))
        # generate new translated rule
        new_rule = Clause(new_rule_atoms[0], new_rule_atoms[1:])
        translated_rules.append(new_rule)

    # append those:
    # target(X):-target_sg1(X).
    # target(X):-target_sg2(X).
    # TODO
    return translated_rules  # , new_lang


def compute_loss(predicted_target_atoms, answer):
    pass


def get_predicted_prob(v_T, answer):
    pass


def train_neumann(
    args,
    NEUMANN,
    I2F,
    optimizer,
    train_loader,
    val_loader,
    test_loader,
    device,
    writer,
    rtpt,
    epochs,
    trial,
):
    bce = torch.nn.BCELoss()
    time_list = []
    iteration = 0
    for epoch in range(epochs):
        loss_i = 0
        start_time = time.time()
        for i, sample in tqdm(enumerate(train_loader, start=0)):
            optimizer.zero_grad()
            # to cuda
            imgs, target_set = map(lambda x: x.to(device), sample)
            target_set = target_set.float()

            # convert the images to probabilistic facts (facts converting)
            V_0 = I2F(imgs)
            # infer and predict the target probability
            V_T = NEUMANN(V_0)
            # get the probabilities of the target atoms
            predicted = get_prob(V_T, NEUMANN, args)
            loss = bce(predicted, target_set)
            loss_i += loss.item()
            # compute the gradients
            loss.backward()
            # update the weights of clauses
            optimizer.step()

            iteration += 1

        # save loss for this epoch
        wandb.log({"metric/training_loss": loss_i})
        epoch_time = time.time() - start_time
        time_list.append(epoch_time)

        rtpt.step()  # subtitle=f"loss={loss_i:2.2f}")
        print("loss: ", loss_i)


def is_in_boxes(pred_box, true_boxes, th=0.7):
    for true_box in true_boxes:
        score = iou_score(pred_box, true_box)
        if score > th:
            return True
    return False


def iou_score(box1, box2):
    return box_iou(box1.unsqueeze(0), box2.unsqueeze(0))[0]


def to_bce_examples(predicted_boxes, predicted_scores, answer_boxes, device):
    predicted_target_probs = []
    true_box_labels = []
    for pred_box, pred_prob in zip(predicted_boxes, predicted_scores):
        if is_in_boxes(pred_box, answer_boxes):
            true_box_labels.append(1.0)
            predicted_target_probs.append(pred_prob.unsqueeze(-1))
        else:
            true_box_labels.append(0.0)
            predicted_target_probs.append(pred_prob.unsqueeze(-1))
    predicted_target_probs_tensor = torch.cat(predicted_target_probs, dim=0).to(device)
    true_box_labels_tensor = torch.tensor(true_box_labels).to(device)
    return predicted_target_probs_tensor, true_box_labels_tensor


def are_all_targets_detected(predicted_boxes, predicted_scores, answer_boxes, th=0.1):
    # false if only low probabilities
    if torch.cat([s.unsqueeze(-1) for s in predicted_scores]).max() < th:
        return 0
    flag = True
    for ans_box in answer_boxes:
        if not is_in_boxes(ans_box, predicted_boxes):
            flag = False
    if flag:
        return 1
    else:
        return 0


# def load_llm_rules(id, complexity):
#     path = ""

#     with open(path, "r"):
