import numpy as np
import torch
import torch.nn.functional as F
from anytree import Node, PreOrderIter, RenderTree
from anytree.search import find_by_attr, findall

from .fol.logic import Clause
from .logic_utils import add_true_atom, add_true_atoms, remove_true_atoms, true


class ClauseGenerator(object):
    """Refinement-based clause generator that holds a tree representation of the generation steps.
    """
    def __init__(self, refinement_generator, root_clauses, th_depth, n_sample):
        self.refinement_generator = refinement_generator
        self.th_depth = th_depth
        self.root_clauses = add_true_atoms(root_clauses)
        self.tree = Node(name="root", clause=Clause(true,[]))
        for c in root_clauses:
            Node(name=str(c), clause=c, parent=self.tree)
            #Node(name=str(nc), clause=nc, =target_node)
        self.n_sample = n_sample
        self.is_root = True
        self.refinement_history = set()
        self.refinement_score_history = set()

    
    def generate(self, clauses, clause_scores):
        clauses_to_refine = self.sample_clauses_by_scores(clauses, clause_scores)
        print("=== CLAUSES TO BE REFINED ===")
        for i, cr in enumerate(clauses_to_refine):
            print(i, ': ', cr)

        self.refinement_history = self.refinement_history.union(set(clauses_to_refine))
        # self.refinement_history = list(set(self.clause_history))
        new_clauses = add_true_atoms(self.apply_refinement(remove_true_atoms(clauses_to_refine)))
        # prune already appeared clauses
        new_clauses = [c for c in new_clauses if not c in self.refinement_history]
        return list(set(new_clauses))
    

    def sample_clauses_by_scores(self, clauses, clause_scores):
        clauses_to_refine = []
        print("Logits for the sampling: ")
        print(np.round(clause_scores.cpu().numpy(), 2))

        n_sampled = 0
        while n_sampled < self.n_sample:
            if len(clauses) == 0:
                # no more clauses to be sampled
                break
            i_sampled_onehot = F.gumbel_softmax(clause_scores,  tau=1.0, hard=True)
            i_sampled = int(torch.argmax(i_sampled_onehot, dim=0).item())
            # selected_clause_indices = [i for i, j in enumerate(selected_clause_indices)]
            # clauses_to_refine_i = [c for i, c in enumerate(clauses) if selected_clause_indices[i] > 0]
            sampled_clause = clauses[i_sampled]
            score = np.round(clause_scores[i_sampled].cpu().numpy(), 2)

            if score in self.refinement_score_history:
                # if a clause with the same score is already sampled, just skip this
                # renormalize clause scores
                if i_sampled != len(clauses)-1:
                    clause_scores = torch.cat([clause_scores[:i_sampled], clause_scores[i_sampled + 1:]])
                    clauses.remove(sampled_clause)
                else:
                    clause_scores = clause_scores[:i_sampled]
                    clauses.remove(sampled_clause)
            else:
                # append to the result
                clauses_to_refine.append(sampled_clause)
                # update history
                self.refinement_score_history.add(score)
                self.refinement_history.add(sampled_clause)
                # renormalize socres
                if i_sampled != len(clauses)-1:
                    clause_scores = torch.cat([clause_scores[:i_sampled], clause_scores[i_sampled + 1:]])
                    clauses.remove(sampled_clause)
                else:
                    clause_scores = clause_scores[:i_sampled]
                    clauses.remove(sampled_clause)

                n_sampled += 1
            
        clauses_to_refine = list(set(clauses_to_refine))
        return clauses_to_refine

    def split_by_head_preds(self, clauses, clause_scores):
        head_pred_clauses_dic = {}
        head_pred_scores_dic = {}
        for i, c in enumerate(clauses):
            if c.head.pred in head_pred_clauses_dic:
                head_pred_clauses_dic[c.head.pred].append(c)
                head_pred_scores_dic[c.head.pred].append(clause_scores[i])
            else:
                head_pred_clauses_dic[c.head.pred] = [c]
                head_pred_scores_dic[c.head.pred] = [clause_scores[i]]
        
        for p in head_pred_scores_dic.keys():
            head_pred_scores_dic[p] = torch.tensor(head_pred_scores_dic[p])
        return head_pred_clauses_dic, head_pred_scores_dic
        
    
    def apply_refinement(self, clauses):
        all_new_clauses = []
        for clause in clauses:
            new_clauses = self.generate_clauses_by_refinement(clause)
            all_new_clauses.extend(new_clauses)
            # add to the refinement tree
            #self.print_tree()
            # the true atom for the clause to be refined has been removed
            target_node = find_by_attr(self.tree, name='clause', value=add_true_atom(clause))
            #print(target_node)
            for nc in new_clauses:
                all_nodes =list(PreOrderIter(self.tree))
                if not nc in [n.clause for n in all_nodes]:
                    Node(name=str(nc), clause=nc, parent=target_node)
                """
                clauses_exist = [n.clause for n in target_node.children]
                if not nc in clauses_exist:
                    print(target_node)
                    print('clauses_exist: ', clauses_exist)
                    Node(name=str(nc), clause=nc, parent=target_node)
                """
                # target_node.children.append(child_node)
        return all_new_clauses
            

    def generate_clauses_by_refinement(self, clause):
        return list(set(add_true_atoms(self.refinement_generator.refine_clause(clause))))

    def print_tree(self):
        print("-- rule generation tree --")
        print(RenderTree(self.tree).by_attr('clause'))

    def get_clauses_by_th_depth(self, th_depth):
        """Get all clauses that are located deeper nodes than given threashold."""
        nodes = findall(self.tree, filter_=lambda node: node.depth >= th_depth)
        return [node.clause for node in nodes]
    """
    def __sample_clauses_by_scores(self, clauses, clause_scores):
        clauses_to_refine = []
        print("Logits for the sampling: ")
        print(np.round(clause_scores.cpu().numpy(), 2))
        for i in range(clause_scores.size(0)):
            clauses_dic, scores_dic = self.split_by_head_preds(clauses, clause_scores[i])
            for p, clauses_p in clauses_dic.items():
                selected_clause_indices = torch.stack([F.gumbel_softmax(scores_dic[p] * 100, tau=1.0, hard=True) for j in range(int(self.n_sample / len(clauses_dic.keys())))])
                selected_clause_indices, _ = torch.max(selected_clause_indices, dim=0)
                # selected_clause_indices = [i for i, j in enumerate(selected_clause_indices)]
                clauses_to_refine_i = [c for i, c in enumerate(clauses_p) if selected_clause_indices[i] > 0]
                clauses_to_refine.extend(clauses_to_refine_i)
            clauses_to_refine = list(set(clauses_to_refine))
        return clauses_to_refine


        def sample_clauses_by_scores(self, clauses, clause_scores):
        clauses_to_refine = []
        print("Logits for the sampling: ")
        print(np.round(clause_scores.cpu().numpy(), 2))
        for i in range(clause_scores.size(0)):
            selected_clause_indices = torch.stack([F.gumbel_softmax(clause_scores[i],  tau=1.0, hard=True) for j in range(int(self.n_sample))])
            selected_clause_indices, _ = torch.max(selected_clause_indices, dim=0)
            # selected_clause_indices = [i for i, j in enumerate(selected_clause_indices)]
            clauses_to_refine_i = [c for i, c in enumerate(clauses) if selected_clause_indices[i] > 0]
            clauses_to_refine.extend(clauses_to_refine_i)
        clauses_to_refine = list(set(clauses_to_refine))
        return clauses_to_refine
    """