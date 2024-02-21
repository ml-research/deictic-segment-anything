import itertools
import multiprocessing

import networkx as nx
import torch
from joblib import Parallel, delayed
from torch_geometric.utils import from_networkx
from tqdm import tqdm

from .fol.logic import Clause, Conjunction
from .fol.logic_ops import subs_list, unify
from .logic_utils import generate_substitutions


class ReasoningGraphModule(object):
    """Reasoning graph, which represents a forward-reasoning process as a bipartite graph.

    Args:
        clauses (list(clauses)): The set of clauses.
        facts (list(atom)): The set of ground atoms (facts).
        terms (list(term)): The set of ground terms.
        lang (language): A language of first-order logic.
        device (device): A device.
        max_term_depth (int): The maximum depth (height) of the term in terms of the functors.
        init (bool): The flag whether the initialization is performed or not.
    """

    def __init__(
        self,
        clauses,
        facts,
        terms,
        lang,
        device,
        max_term_depth,
        init=True,
        clause_casche=set(),
        grounding_casche={},
    ):
        self.lang = lang
        self.clauses = clauses
        self.facts = facts
        self.fact_set = set(facts)
        self.terms = terms
        self.device = device
        self.max_term_depth = max_term_depth
        self.grounding_casche = grounding_casche
        self.clause_casche = clause_casche
        if init:
            self.fact_index_dict = self._build_fact_index_dict(facts)
            self.grounded_clauses, self.clause_indices = self._ground_clauses(
                clauses, lang
            )
            # for i, clause in enumerate(self.grounded_clauses):
            # print(i, clause)
            self.atom_node_idxs = list(range(len(self.facts)))
            self.conj_node_idxs = list(
                range(len(self.facts), len(self.facts) + len(self.grounded_clauses) + 1)
            )  # a dummy conj node
            # print('Building reasoning graph for {}'.format(str(clauses)))
            # build reasoning graph
            self.networkx_graph, self.node_labels, self.node_objects = self._build_rg()
            # print("Converting to PyG object ...")
            self.pyg_data = from_networkx(self.networkx_graph)
            self.edge_index = self.pyg_data.edge_index.to(device)
            self.edge_type = torch.tensor(self.pyg_data.etype).to(device)
            self.edge_clause_index = torch.tensor(self.pyg_data.clause_index).to(device)
            self.num_nodes = len(self.node_labels)

    def _build_fact_index_dict(self, facts):
        dic = {}
        for i, fact in enumerate(facts):
            dic[fact] = i
        return dic

    def __str__(self):
        N_atom_nodes = len(self.atom_node_idxs)
        N_conj_nodes = len(self.conj_node_idxs)
        return "Reasoning Graph(N_atom_nodes={}, N_conj_nodes={})".format(
            N_atom_nodes, N_conj_nodes
        )

    def __repr__(self):
        return self.__str__()

    def _get_fact_idx(self, fact):
        if not fact in self.fact_index_dict:
            return False, -1
        else:
            return True, self.fact_index_dict[fact]

    def _invalid_var_dtypes(self, var_dtypes):
        # check the contradiciton of the List[(var, dtype)]
        if len(var_dtypes) < 2:
            return False
        for i in range(len(var_dtypes) - 1):
            for j in range(i, len(var_dtypes)):
                if (
                    var_dtypes[i][0] == var_dtypes[j][0]
                    and var_dtypes[i][1] != var_dtypes[j][1]
                ):
                    return True
        return False

    def _ground_clauses(self, clauses, lang):
        """
        Ground a clause using all of the constants in a language.

        Args:
            clause (Clause): A clause.
            lang (Language): A language.
        """

        ground_clauses_list = Parallel(n_jobs=20)(
            [delayed(self._ground_clause)(clause) for clause in clauses]
        )
        # update casche
        # for i, c in enumerate(clauses):
        #    self.clause_casche.add(c)
        #    self.grounding_casche[str(c)] = ground_clauses_list[i]

        all_ground_clauses = []
        all_clause_indices = []
        for clause_index, ground_clauses in enumerate(ground_clauses_list):
            indices = [clause_index] * len(ground_clauses)
            all_ground_clauses.extend(ground_clauses)
            all_clause_indices.extend(indices)

        return all_ground_clauses, all_clause_indices

    def _ground_clause(self, clause):
        """Produce all ground clauses given a clause."""
        if len(clause.all_vars()) == 0:
            # print("Grounding completed with {} substitutions!: {}".format(0, str(clause)))
            return [clause]
        else:
            theta_list = generate_substitutions(
                [clause.head] + clause.body, self.terms, self.max_term_depth
            )
            ground_clauses = []
            for i, theta in enumerate(theta_list):
                ground_head = subs_list(clause.head, theta)
                if ground_head in self.fact_index_dict:
                    ground_body = [subs_list(bi, theta) for bi in clause.body]
                    ground_clauses.append(Clause(ground_head, ground_body))

            ground_clauses = self._remove_redundunt_ground_clauses(ground_clauses)
            return ground_clauses

    def _build_rg(self):
        """Build reasoning graph from clauses."""

        # print('Building Reasoning Graph...')
        G, node_labels, node_objects = self._init_rg()
        edge_clause_index = []

        # add dummy edge T to T
        G.add_edge(
            0,
            len(self.facts) + len(self.grounded_clauses),
            etype=0,
            color="r",
            clause_index=0,
        )
        G.add_edge(
            len(self.facts) + len(self.grounded_clauses),
            0,
            etype=1,
            color="r",
            clause_index=0,
        )

        # print("Adding edges to the graph...")
        for i, gc in enumerate(self.grounded_clauses):
            head_flag, head_fact_idx = self._get_fact_idx(gc.head)
            body_fact_idxs = []
            body_flag = True
            for bi in gc.body:
                body_flag, body_fact_idx = self._get_fact_idx(bi)
                body_fact_idxs.append(body_fact_idx)
                if not body_flag:
                    # failed to find body fact in database
                    break
            if body_flag and head_flag:
                head_node_idx = head_fact_idx
                body_node_idxs = body_fact_idxs
                conj_node_idx = self.conj_node_idxs[i]
                for body_node_idx in body_node_idxs:
                    G.add_edge(
                        conj_node_idx,
                        head_node_idx,
                        etype=1,
                        color="r",
                        clause_index=self.clause_indices[i],
                    )
                    G.add_edge(
                        body_node_idx,
                        conj_node_idx,
                        etype=0,
                        color="b",
                        clause_index=self.clause_indices[i],
                    )
        return G, node_labels, node_objects

    def _remove_redundunt_ground_clauses(self, ground_clauses):
        """Rmove A:-B_1,...,B_n if A or B_i is not in the fact list, this is redundant ground clause not effective to the reasoning result."""
        pruned_ground_clauses = []
        for gc in ground_clauses:
            if gc.head in self.fact_set:
                body_flag = True
                for bi in gc.body:
                    if not bi in self.fact_set:
                        body_flag = False
                if body_flag:
                    pruned_ground_clauses.append(gc)
        return pruned_ground_clauses

    def _to_edge_index(self, clauses):
        """NOT USED IN THIS VERSION.
        Convert clauses into a edge index representing a reasoning graph.

        Args:
            clauses (list(Clauses)): A set of clauses.
        """
        gc_counter = 0
        edge_index = []
        edge_type = []
        edge_clause_index = []
        for i, clause in enumerate(clauses):
            grounded_clauses = self._ground_clause(clause)
            for gc in grounded_clauses:
                head_fact_idx = self.facts.index(gc.head)
                conj_idx = len(self.facts) + gc_counter
                body_fact_idxs = []
                for bi in gc.body:
                    body_fact_idx = self.facts.index(bi)
                    # G.add_edge(body_node_idx, conj_node_idx, etype=0, color='b')
                    new_edge = [body_fact_idx, conj_idx]
                    if not new_edge in edge_index:
                        edge_index.append([body_fact_idx, conj_idx])
                        edge_type.append(0)  # edge: atom_node -> conj_node
                        edge_clause_index.append(i)

                # G.add_edge(conj_node_idx, head_node_idx, etype=1, color='r')
                edge_index.append([conj_idx, head_fact_idx])
                edge_type.append(1)  # edge: conj_node -> atom_node
                edge_clause_index.append(i)
                gc_counter += 1
        edge_index = torch.tensor(edge_index).view((2, -1)).to(self.device)
        edge_clause_index = torch.tensor(edge_clause_index).to(self.device)
        edge_type = torch.tensor(edge_type).to(self.device)
        num_nodes = conj_idx

        # compute indicis for atom nodes and conjunction nodes
        self.atom_node_idxs = list(range(len(self.facts)))
        self.conj_node_idxs = list(range(len(self.facts), len(self.facts) + gc_counter))
        return edge_index, edge_clause_index, edge_type, num_nodes

    def _init_rg(self):
        # print("Initializing reasoning graph...")
        G = nx.DiGraph()
        node_labels = {}
        node_objects = {}
        # atom_idxs = self.node_idx_dic['atom']
        # conj_idxs = self.node_idx_dic['conj']
        # print("Initializing the reasoning graph...")
        G.add_nodes_from(list(range(len(self.facts))))

        N_fact = len(self.facts)
        for i, fact in enumerate(self.facts):
            node_labels[i] = str(fact)
            node_objects[i] = fact
        # G.add_nodes_from(conj_idxs)
        G.add_nodes_from(list(range(N_fact, N_fact + len(self.grounded_clauses))))
        for i, conj in enumerate(self.grounded_clauses):
            node_labels[N_fact + i] = "∧"
            node_objects[N_fact + i] = Conjunction()

        # add dummy conj node and edge T -> dummy_con
        G.add_node(N_fact + len(self.grounded_clauses))
        node_labels[N_fact + len(self.grounded_clauses)] = "∧"
        node_objects[N_fact + len(self.grounded_clauses)] = Conjunction()

        return G, node_labels, node_objects
