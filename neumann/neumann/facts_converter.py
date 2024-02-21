import torch
import torch.nn as nn
from tqdm import tqdm

from .fol.logic import NeuralPredicate


class FactsConverter(nn.Module):
    """FactsConverter converts the output fromt the perception module to the valuation vector.
    """

    def __init__(self, lang, atoms, bk, device=None):
        super(FactsConverter, self).__init__()
        # self.e = perception_module.e
        # self.d = perception_module.d
        self.lang = lang
        # self.vm = valuation_module  # valuation functions
        self.device = device
        self.atoms = atoms
        self.bk = bk
        # init indices
        self.np_indices = self._get_np_atom_indices()
        self.bk_indices = self._get_bk_atom_indices()

    def __str__(self):
        return "FactsConverter(entities={}, dimension={})".format(self.e, self.d)

    def __repr__(self):
        return "FactsConverter(entities={}, dimension={})".format(self.e, self.d)

    def _get_np_atom_indices(self):
        """Pre compute the indices of atoms with neural predicats."""
        indices = []
        for i, atom in enumerate(self.atoms):
            if type(atom.pred) == NeuralPredicate:
                indices.append(i)
        return indices

    def _get_bk_atom_indices(self):
        """Pre compute the indices of atoms in background knowledge."""
        indices = []
        for i, atom in enumerate(self.atoms):
            if atom in self.bk:
                indices.append(i)
        return indices

    def forward(self, Z):
        return self.convert(Z)

    def get_params(self):
        return self.vm.get_params()

    def init_valuation(self, n, batch_size):
        v = torch.zeros((batch_size, n)).to(self.device)
        v[:, 1] = 1.0
        return v

    def filter_by_datatype():
        pass

    def to_vec(self, term, zs):
        pass

    
    def convert(self, bk_atoms=None):
        V = torch.zeros(1, len(self.atoms)).to(self.device)

        # add background knowledge
        for i, atom in enumerate(self.atoms):
            if atom in bk_atoms:
                V[0, i] += 1.0

        V[0, 0] += 1.0
        return V


class FactsConverterWithQuery(nn.Module):
    """FactsConverter converts the output fromt the perception module to the valuation vector.
    """

    # def __init__(self, lang, perception_module, valuation_module, device=None):
    def __init__(self, lang, atoms, bk, perception_module, valuation_module, device=None):
        super(FactsConverterWithQuery, self).__init__()
        self.e = perception_module.e
        self.d = perception_module.d
        self.lang = lang
        self.vm = valuation_module  # valuation functions
        self.device = device
        self.atoms = atoms
        self.bk = bk
        # init indices
        self.np_indices = self._get_np_atom_indices()
        self.bk_indices = self._get_bk_atom_indices()

    def __str__(self):
        return "FactsConverter(entities={}, dimension={})".format(self.e, self.d)

    def __repr__(self):
        return "FactsConverter(entities={}, dimension={})".format(self.e, self.d)

    def _get_np_atom_indices(self):
        """Pre compute the indices of atoms with neural predicats."""
        indices = []
        for i, atom in enumerate(self.atoms):
            if type(atom.pred) == NeuralPredicate:
                indices.append(i)
        return indices

    def _get_bk_atom_indices(self):
        """Pre compute the indices of atoms in background knowledge."""
        indices = []
        for i, atom in enumerate(self.atoms):
            if atom in self.bk:
                indices.append(i)
        return indices

    def forward(self, Z, Q):
        return self.convert(Z, Q)

    def get_params(self):
        return self.vm.get_params()

    def init_valuation(self, n, batch_size):
        v = torch.zeros((batch_size, n)).to(self.device)
        v[:, 1] = 1.0
        return v

    def filter_by_datatype():
        pass

    def to_vec(self, term, zs):
        pass

    def convert(self, Z, Q):
        batch_size = Z.size(0)

        # V = self.init_valuation(len(G), Z.size(0))
        #V = torch.zeros((batch_size, len(G))).to(
        #    torch.float32).to(self.device)
        V = torch.zeros((batch_size, len(self.atoms))).to(
            torch.float32).to(self.device)

        # T to be 1.0
        V[:, 0] = 1.0

        for i in self.np_indices:
                V[:, i] = self.vm(Z, Q, self.atoms[i])

        for i in self.bk_indices:
                V[:, i] += torch.ones((batch_size, )).to(
                    torch.float32).to(self.device)
        """
        for i, atom in enumerate(G):
            if type(atom.pred) == NeuralPredicate:
                V[:, i] = self.vm(Z, Q, atom)
            elif atom in B:
                # V[:, i] += 1.0
                V[:, i] += torch.ones((batch_size, )).to(
                    torch.float32).to(self.device)
        """
        return V