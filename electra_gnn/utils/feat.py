from rdkit import Chem
from rdkit.Chem import rdmolfiles
from rdkit.Chem import rdmolops
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import torch


def featurize(smiles: str) -> Data:
    """Featurize a molecule, each atom only uses atomic number

    Args:
        smiles (str): SMILES of the molecule

    Returns:
        [torch_geometric.data.Data]: torch geometric Data entry
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:  # when rdkit fails to read a molecule it returns None
        return None
    new_order = rdmolfiles.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, new_order)

    edges = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append([start, end])
    if not edges:  # If no edges (bonds) were found, exit (single ion etc)
        return None
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(atom.GetAtomicNum())

    x = torch.tensor(atom_features, dtype=torch.long)
    edge_index = torch.tensor(edges).t().contiguous()

    return Data(
        x=x,
        edge_index=to_undirected(edge_index, num_nodes=x.size(0)),
    )
