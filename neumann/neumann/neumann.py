import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from .logic_utils import add_true_atoms, get_index_by_predname, true
from .torch_utils import softor


class NEUMANN(nn.Module):
    """The Neuro-Sybolic Message Passing Reasoner (NEUMANN)  ʕ•̀ω•́ʔ✧

    Args:
        clauses (list(clauses)): The set of clauses.
        atoms (list(atom)): The set of ground atoms (facts).
        message_passing_module (.message_passing.MessagePassingReasoner): The message passing module.
        reasoning_graph_module (.reasoning_graph.ReasoningGraphModule): The reasoning graph module.
        program_size (int): The number of the clauses to be chosen e.g. 5 clauses out of 10 clauses (program_size=5).
        device (device): A device.
        bk_clauses (list(clauses)): The set of clauses as background knowledge with fixed weights.
        train (bool): The flag to be trained or not.
    """

    def __init__(
        self,
        clauses,
        atoms,
        message_passing_module,
        reasoning_graph_module,
        program_size,
        device,
        bk=None,
        bk_clauses=None,
        train=False,
        softmax_tmp=1.0,
        explain=False,
    ):
        super().__init__()
        self.atoms = atoms
        self.atom_strs = [str(atom) for atom in self.atoms]
        self.clauses = add_true_atoms(clauses)
        self.bk = bk
        self.bk_clauses = add_true_atoms(bk_clauses)
        self.mpm = message_passing_module
        self.rgm = reasoning_graph_module
        self.program_size = program_size
        self.softmax_temp = softmax_tmp
        print(self.rgm)
        self.train = train
        self.clause_weights_bk = self.get_ones_weights(bk_clauses, device)
        if train:
            self.init_random_weights(program_size, clauses, device)
            print(self.clause_weights)
            print(clauses)
        else:
            self.init_ones_weights(clauses, device)
        self.device = device
        self.explain = explain
        # self.print_program()

    def init_ones_weights(self, clauses, device):
        """Initialize the clause weights with fixed weights. All clauses are asuumed to be correct rules."""
        # self.clause_weights = nn.Parameter(
        #     torch.ones((len(clauses),), dtype=torch.float32).to(device)
        # )
        self.clause_weights = torch.ones((len(clauses),), dtype=torch.float32).to(
            device
        )

    def get_ones_weights(self, clauses, device):
        """Initialize the clause weights with fixed weights. All clauses are asuumed to be correct rules."""
        return torch.ones((len(clauses),), dtype=torch.float32).to(device)

    def init_random_weights(self, program_size, clauses, device):
        """Initialize the clause weights with a random initialization."""
        # self.clause_weights = nn.Parameter(
        #     torch.Tensor(np.random.rand(program_size, len(clauses))).to(device)
        # )
        self.clause_weights = torch.Tensor(
            np.random.rand(program_size, len(clauses))
        ).to(device)

    def _softmax_clauses(self, clause_weights):
        """Take softmax of clause weights to choose M clauses."""
        clause_weights_sm = torch.softmax(clause_weights, dim=1)
        return softor(clause_weights_sm, dim=0)

    def _get_clause_weights(self):
        """Compute clause_weights by taking softmax and softor"""
        if not self.train:
            clause_weights = self.clause_weights
        else:
            clause_weights = softor(
                torch.softmax(self.clause_weights / self.softmax_temp, dim=1), dim=0
            )

        # concatenate bk_clauses if it exists
        if self.bk_clauses != None:
            clause_weights_bk = self.clause_weights_bk
            return torch.cat([clause_weights, clause_weights_bk])
        else:
            return clause_weights  # [0.9, 0.1, 0.8, 1.0, ..., 1.0]

    def _preprocess_clauses(self, clauses):
        """Add T (true) to the clause body which has no body atoms e.g. p(X,Y):-. => p(X,Y):-T."""
        cs = []
        for clause in clauses:
            if len(clause.body) == 0:
                clause.body = [true]
                cs.append(clause)
            else:
                cs.append(clause)
        return cs

    def _print_graph(self):
        for i, index_pair in enumerate(self.edge_index.view(-1, 2)):
            i1 = index_pair[0].detach().numpy()
            i2 = index_pair[1].detach().numpy()
            clause_index = self.edge_clause_index[i].detach().numpy()
            print("Clause: {}".format(self.clauses[clause_index]))
            if i1 < len(self.atoms):
                print("Edge {} -> {}".format(str(self.atoms[i1]), "conj"))
            elif i2 < len(self.atoms):
                print("Edge {} -> {}".format("conj", str(self.atoms[i2])))

    def _get_edge_type(self, G):
        v = []
        for i, (x, y, attr) in enumerate(G.edges.data()):
            if attr["etype"] == "atom":
                v.append(0)
            elif attr["etype"] == "conj":
                v.append(1)
            else:
                assert True, "Invalid edge type."
        return torch.tensor(v)

    def get_params(self):
        # print('GNN params: ', len(list(self.gnn.parameters())))
        return (
            list(self.gnn.parameters())
            + list(self.node_out.parameters())
            + self.nfm.get_params()
        )  # + list(self.nfm.get_params())# + list(self.pm.get_params())

    def _flatten(self, x, batch_size):
        return x.unsqueeze(0).view(batch_size, -1)

    def _to_attribute_matrix(self, x):
        """Convert a valuation into an attribute matrix (data.x) by padding with zeros for conjunction nodes."""
        batch_size = x.size(0)
        num_conj_nodes = len(self.rgm.conj_node_idxs)
        zeros = torch.zeros((batch_size, num_conj_nodes, 1), dtype=torch.float).to(
            self.device
        )

        if self.bk != None:
            for bk_i in self.bk:
                # substitute 1.0 for bk atoms
                x[:, self.atoms.index(bk_i)] = 1.0
        # B * N_node * 1
        return torch.cat([x, zeros], dim=1)

    def forward(self, x):
        """Forwarding function for NEUMANN. It proceeds as follows:
        (1) Computes the node features (probabilities) from a set of probabilistic facts (valuation vector).
        (2) Transforms the node features and a reasoning graph into a graph data (as a batch).
        (3) Computes the indices of atom nodes and conjunction nodes for a graph data extended for the batch.
        (4) Performs the bi-directional message-passing on the graph data.
        Args:
            x (Tensor): A batch of valuation vectors (probabilities of atoms).
        Return:
            y (Tensor): A batch of valuation vectors (probabilities of atoms and conjunctions) after reasoning.
        """
        batch_size = x.size(0)

        # convert probabilistic facts to a node-feature matrix
        x = self._to_attribute_matrix(x.unsqueeze(-1))

        # transform the matrix to a batch of data (a graph representing the graphs)
        x = self._to_batch_data(x, batch_size).to(self.device)

        # get indices to message passing
        atom_node_idxs, conj_node_idxs = self._get_idxs(batch_size)

        clause_weights = self._get_clause_weights()

        # forwarding to message passing reasoning module
        y = self.mpm(
            x,
            clause_weights,
            self.rgm.edge_clause_index,
            self.rgm.edge_type,
            atom_node_idxs,
            conj_node_idxs,
            batch_size,
            explain=self.explain,
        )
        return y

    def _get_idxs(self, batch_size):
        """Increment the indicies for batch computation."""
        atom_node_idxs = torch.tensor(self.rgm.atom_node_idxs).to(self.device)
        conj_node_idxs = torch.tensor(self.rgm.conj_node_idxs).to(self.device)

        num_nodes = torch.tensor(self.rgm.num_nodes).to(self.device)

        atom_node_idxs_batch = torch.cat(
            [atom_node_idxs + i * num_nodes for i in range(batch_size)]
        )
        conj_node_idxs_batch = torch.cat(
            [conj_node_idxs + i * num_nodes for i in range(batch_size)]
        )
        return atom_node_idxs_batch, conj_node_idxs_batch

    def _to_batch_data(self, x, batch_size):
        """Generate the batched data for GNN.
        Args:
            x (tensor): B * N_node * dim_node
            edge_index (tensor): An edge_index tensor.
        Returns:
            a batch of data (torch_geometric.data.Batch)
        """
        data_list = []
        for i in range(batch_size):
            data = Data(x=x[i], edge_index=self.rgm.edge_index)
            data_list.append(data)
        return Batch.from_data_list(data_list)

    def predict(self, x):
        """Get atom probs."""
        # v: batch * |atoms| * N_features

        x = x[:, : len(self.atoms)]
        x = self.predict_node(x)
        return x

    def predict_by_predname(self, x, predname):
        """Extracting a value from the valuation tensor using a given predicate."""
        # v: batch * |atoms|
        target_index = get_index_by_predname(pred_str=predname, atoms=self.atoms)
        return x[:, target_index]

    def predict_by_atom(self, x, atom_str):
        atom_index = self.atom_strs.index(atom_str)
        return x[:, atom_index]

    def predict_node(self, x):
        return torch.cat([self.node_out(x[:, i, :]) for i in range(x.size(1))], dim=1)

    def predict_multi(self, v, prednames):
        """Extracting values from the valuation tensor using given predicates.
        Example: prednames = ['kp1', 'kp2', 'kp3']
        """
        # v: batch * |atoms|
        target_indices = []
        for predname in prednames:
            target_index = get_index_by_predname(pred_str=predname, atoms=self.atoms)
            target_indices.append(target_index)
        prob = torch.cat([v[:, i].unsqueeze(-1) for i in target_indices], dim=1)
        B = v.size(0)
        N = len(prednames)
        assert (
            prob.size(0) == B and prob.size(1) == N
        ), "Invalid shape in the prediction."
        return prob

    def print_program(self):
        """Print asummary of logic programs using continuous weights."""
        print("===== LOGIC PROGRAM =====")
        clause_weights = self._get_clause_weights()
        clauses = self.clauses + self.bk_clauses
        for i, w in enumerate(clause_weights):
            print(
                "C_" + str(i) + ": ", np.round(w.detach().cpu().numpy(), 2), clauses[i]
            )

    def print_valuation_batch(self, valuation, n=40):
        for b in range(valuation.size(0)):
            print("===== BATCH: ", b, "=====")
            v = valuation[b].detach().cpu().numpy()
            idxs = np.argsort(-v)
            for i in idxs:
                if v[i] > 0.2:
                    if (
                        not self.atoms[i].pred.name
                        in [
                            "member",
                            "not_member",
                            "get_color",
                            "perm",
                            "delete",
                            "right_most",
                        ]
                        and not self.atoms[i].pred.name
                        in ["first_obj", "second_obj", "third_obj", "append", "reverse"]
                        and not self.atoms[i].pred.name
                        in ["left_of", "same_position", "smaller", "chain"]
                    ):
                        print(i, self.atoms[i], ": ", round(v[i], 3))

    def print_trace_batch(self, valuation_list, n=40):
        for i, valuation in enumerate(valuation_list):
            print("Step {}:".format(i))
            if i == 0:
                self.print_valuation_batch(valuation.unsqueeze(0).squeeze(-1))
            else:
                vs = valuation - valuation_before
                vs = vs.unsqueeze(0).squeeze(-1)
                for b in range(vs.size(0)):
                    print("===== BATCH: ", b, "=====")
                    v = vs[b].detach().cpu().numpy()
                    idxs = np.argsort(-v)
                    for j in idxs:
                        if v[j] > 0.1:
                            print(j, self.atoms[j], ": ", round(v[j], 3))
            valuation_before = valuation

    def get_valuation_text(self, valuation):
        text_batch = ""  # texts for each batch
        for b in range(valuation.size(0)):
            top_atoms = self.get_top_atoms(valuation[b].detach().cpu().numpy())
            top_atoms = [
                atom
                for atom in top_atoms
                if not atom
                in ["member", "not_member", "delete", "diff_color", "right_most"]
            ]
            text = "----BATCH " + str(b) + "----\n"
            text += self.atoms_to_text(top_atoms)
            text += "\n"
            # texts.append(text)
            text_batch += text
        return text_batch

    def get_top_atoms(self, v, th=0.7):
        top_atoms = []
        for i, atom in enumerate(self.atoms):
            if v[i] > th:
                top_atoms.append(atom)
        return top_atoms

    def get_top_atoms_with_scores(self, v, th=0.7):
        top_atoms = []
        scores = []
        for i, atom in enumerate(self.atoms):
            if v[i] > th:
                top_atoms.append(atom)
                scores.append(v[i])
        return top_atoms, scores

    def atoms_to_text(self, atoms):
        text = ""
        for atom in atoms:
            text += str(atom) + ", "
        return text

    def to_clingo_text_batch(self, V_0):
        text_list = []
        for v in V_0:
            text = ""
            for clause in self.clauses + self.bk_clauses:
                text += str(clause)
                text += " "
            for i, atom in enumerate(self.atoms):
                if v[i] > 0.5 and not str(atom) == ".(__T__)":
                    text += str(atom)
                    text += ". "
                    # t ext += '.\n'
            text_list.append(text)
        return text_list
