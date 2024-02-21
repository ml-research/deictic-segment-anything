import torch
import torch.nn as nn
import torch_geometric.nn
from torch import Tensor
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import MessagePassing
from torch_scatter import gather_csr, scatter, segment_csr

# from neural_utils import MLP
from .scatter import *


class Atom2ConjConv(MessagePassing):
    """Message Passing class for messages from atom nodes to conjunction nodes.

    Args:
        soft_logic (softlogic): An implementation of the soft-logic operations.
        device (device): A device.
    """

    def __init__(self, soft_logic, device):
        super().__init__(aggr='add')
        self.soft_logic = soft_logic
        self.device = device

    def forward(self, x, edge_index, conj_node_idxs, batch_size):
        """Perform atom2conj message-passing.
        Args:
            x (tensor): A data (node features).
            edge_index (tensor): An edge index.
            conj_node_idxs (tensor): A list of indicies of conjunction nodes extended for the batch size.
            batch_size (int): A batch size.
        """
        return self.propagate(edge_index, x=x, conj_node_idxs=conj_node_idxs, batch_size=batch_size).view((batch_size, -1))

    def message(self, x_j):
        """Compute the message.
        """
        return x_j

    def update(self, message, x, conj_node_idxs):
        """Update the node features. 
        Args:
            message (tensor, [node_num, node_dim]): Messages aggregated by `aggregate`.
            x (tensor, [node_num, node_dim]): Node features on the previous step.
        """
        return self.soft_logic._or(torch.stack([x[conj_node_idxs], message[conj_node_idxs]]))

    def aggregate(self, inputs, index):
        """Aggregate the messages.
        Args:
            inputs (tensor, [num_edges, num_features]): The values come from each edge. 
            index (tensor, [2, num_of_edges]): The indices of the terminal nodes (conjunction nodes).
        """
        return scatter_mul(inputs, index, dim=0)


class Conj2AtomConv(MessagePassing):
    """Message Passing class for messages from atom nodes to conjunction nodes.

    Args:
        soft_logic (softlogic): An implementation of the soft-logic operations.
        device (device): A device.
    """

    def __init__(self, soft_logic, device):
        super().__init__(aggr='add')
        self.soft_logic = soft_logic
        self.device = device
        self.eps = 1e-4
        # self.linear = nn.Linear()

    def forward(self, x, edge_index, edge_weight, edge_clause_index, atom_node_idxs, n_nodes, batch_size):
        """Perform conj2atom message-passing.
            Args:
                edge_index (tensor): An edge index.
                x (tensor): A data (node features).
                edge_weight (tensor): The edge weights.
                edge_clause_index (tensor): A list of indices of clauses representing which clause produced each edge in the reasoning graph.
                atom_node_idxs (tensor): A list of indicies of atom nodes extended for the batch size.
                n_nodes (int): The number of nodes in the reasoning graph.
                batch_size (int): A batch size.
        """
        return self.propagate(edge_index, x=x, edge_weight=edge_weight, edge_clause_index=edge_clause_index,
                              atom_node_idxs=atom_node_idxs, n_nodes=n_nodes, batch_size=batch_size).view(batch_size, -1)

    def message(self, x_j, edge_weight, edge_clause_index):
        """Compute the message.
        """
        return edge_weight.view(-1, 1) * x_j

    def update(self, message, x, atom_node_idxs):
        """Update the node features. 
        Args:
            message (tensor, [node_num, node_dim]): Messages aggregated by `aggregate`.
            x (tensor, [node_num, node_dim]): Node features on the previous step.
        """
        return self.soft_logic._or(torch.stack([x[atom_node_idxs], message[atom_node_idxs]]))

    def aggregate(self, inputs, index, n_nodes):
        """Aggregate the messages.
        Args:
            inputs (tensor, [num_edges, num_features]): The values come from each edge. 
            index (tensor, [2, num_of_edges]): The indices of the terminal nodes (conjunction nodes).
            n_nodes (int): The number of nodes in the reasoning graph.
        """
        # softor
        # gamma = 0.05
        gamma = 0.015
        log_sum_exp = gamma * \
            self._logsumexp((inputs) * (1/gamma), index, n_nodes)
        if log_sum_exp.max() > 1.0:
            return log_sum_exp / log_sum_exp.max()
        else:
            return log_sum_exp

    def _logsumexp(self, inputs, index, n_nodes):
        return torch.log(scatter(src=torch.exp(inputs), index=index, dim=0, dim_size=n_nodes, reduce='sum') + self.eps)


class MessagePassingModule(torch.nn.Module):
    """The bi-directional message-passing module.

    Args:
        soft_logic (softlogic): An implementation of the soft-logic operations.
        device (device): A device.
        T (int): The number of steps for reasoning.
    """

    def __init__(self, soft_logic, device, T):
        super().__init__()
        self.soft_logic = soft_logic
        self.device = device
        self.T = T
        self.atom2conj = Atom2ConjConv(soft_logic, device)
        self.conj2atom = Conj2AtomConv(soft_logic, device)

    def forward(self, data, clause_weights, edge_clause_index, edge_type, atom_node_idxs, conj_node_idxs, batch_size, explain=False):
        """
        Args:
            data (torch_geometric.Data): A logic progam and probabilistic facts as a graph data.
            clause_weights (torch.Tensor): Weights for clauses.
            edge_clause_index (torch.Tensor): A clause indices for each edge, representing which clause produces the edge.
            edge_type (torch.Tensor): Edge types (atom2conj:0 or conj2atom:1)
            atom_node_idxs (torch.Tensor): The indices of atom nodes.
            conj_node_idxs (torch.Tensor): The indices of conjunction nodes.
            batch_size (int): The batch size of input.
        """
        x_atom = data.x[atom_node_idxs]
        x_conj = data.x[conj_node_idxs]


        # filter the edge index using the edge type obtaining the set of atom2conj edges and the set of conj2atom edges
        atom2conj_edge_index = self._filter_edge_index(
            data.edge_index, edge_type, 'atom2conj', batch_size).to(self.device)
        conj2atom_edge_index = self._filter_edge_index(
            data.edge_index, edge_type, 'conj2atom', batch_size).to(self.device)

        # filter the edge-clause index using the edge type obtaining the set of atom2conj edge-clause indices and the set of conj2atom edge-clause indices
        conj2atom_edge_clause_index = self._filter_edge_clause_index(
            edge_clause_index, edge_type, batch_size).to(self.device)
        edge_weight = torch.gather(
            input=clause_weights, dim=0, index=conj2atom_edge_clause_index)

        n_nodes = data.x.size(0)

        self.x_atom_list = [data.x[atom_node_idxs].view((batch_size, -1)).detach().cpu().numpy()[:,1:]]
        x = data.x


        # dummy variable to compute inpute gradients
        if explain:
            #print(x[atom_node_idxs], x[atom_node_idxs].shape)
            self.dummy_zeros = torch.zeros_like(x[atom_node_idxs], requires_grad=True).to(torch.float32).to(self.device)
            self.dummy_zeros.requires_grad_()
            self.dummy_zeros.retain_grad()
            #print(self.dummy_zeros)
            # add dummy zeros to get input gradients
            x[atom_node_idxs] = x[atom_node_idxs] + self.dummy_zeros

        # iterate message passing T times
        for t in range(self.T):

            # step 1: Atom -> Conj
            x_conj_new = self.atom2conj(
                x, atom2conj_edge_index, conj_node_idxs, batch_size)

            # create new tensor (node features) by updating conjunction embeddings
            x = self._cat_x_atom_x_conj(
                x[atom_node_idxs].view((batch_size, -1)), x_conj_new)

            # step 2: Conj -> Atom
            x_atom_new = self.conj2atom(x=x, edge_weight=edge_weight, edge_index=conj2atom_edge_index, edge_clause_index=edge_clause_index,  atom_node_idxs=atom_node_idxs,
                                        n_nodes=n_nodes, batch_size=batch_size)
            self.x_atom_list.append(x_atom_new.detach().cpu().numpy()[:,1:])

            x = self._cat_x_atom_x_conj(x_atom_new, x_conj_new)
        self.x_atom_final = x_atom_new
        return x_atom_new

    def _cat_x_atom_x_conj(self, x_atom, x_conj):
        """Concatenate the features of atom ndoes and those of conj nodes.
        Args:
            x_atom : batch_size * n_atom
            x_conj : batch_size * n_conj

        Returns:
            [x_atom_1, x_conj_1, x_atom_2, x_conj_2, ...]
        """
        xs = []
        for i in range(x_atom.size(0)):
            x_i = torch.cat([x_atom[i], x_conj[i]])
            xs.append(x_i)
        return torch.cat(xs).unsqueeze(-1)

    def _filter_edge_clause_index(self, edge_clause_index, edge_type, batch_size):
        """Filter the edge index by the edge type.
        """
        edge_clause_index = torch.stack(
            [edge_clause_index for i in range(batch_size)]).view((-1))
        edge_type = torch.stack([edge_type for i in range(batch_size)])
        mask = (edge_type == 1).view((-1))
        return edge_clause_index[mask]

    def _filter_edge_index(self, edge_index, edge_type, mode, batch_size):
        """Filter the edge index by the edge type.
        """
        edge_type = torch.stack([edge_type for i in range(batch_size)])
        if mode == 'atom2conj':
            mask = (edge_type == 0).view((-1))
            return edge_index[:, mask]
        elif mode == 'conj2atom':
            mask = (edge_type == 1).view((-1))
            return edge_index[:, mask]
        else:
            assert 0, "Invalid mode in _filter_edge_index"
