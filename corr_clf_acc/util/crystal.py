import numpy
import pandas
import torch
from tqdm import tqdm
from pymatgen.core.structure import Structure
from torch_geometric.data import Data
from util.base import get_elem_feats


def rbf(x, mu, beta):
    return numpy.exp(-(x - mu)**2 / beta**2)


def load_dataset(path, metadata_file, idx_target, n_bond_feats=32, radius=4):
    elem_feats = get_elem_feats()
    list_cgs = list()
    metadata = numpy.array(pandas.read_excel(metadata_file))
    metadata = numpy.hstack([metadata, numpy.arange(metadata.shape[0]).reshape(-1, 1)])
    targets = metadata[:, idx_target]

    for i in tqdm(range(0, metadata.shape[0])):
        cg = read_cif(elem_feats, path, str(metadata[i, 0]), n_bond_feats, radius, targets[i])

        if cg is not None:
            cg.gid = len(list_cgs)
            list_cgs.append(cg)

    return list_cgs


def read_cif(elem_feats, path, m_id, n_bond_feats, radius, target):
    crys = Structure.from_file(path + '/' + m_id + '.cif')
    atom_feats = get_atom_feats(crys, elem_feats)
    bonds, bond_feats = get_bonds(crys, n_bond_feats, radius)

    if bonds is None:
        return None

    atom_feats = torch.tensor(atom_feats, dtype=torch.float).cuda()
    bonds = torch.tensor(bonds, dtype=torch.long).t().contiguous().cuda()
    bond_feats = torch.tensor(bond_feats, dtype=torch.float).cuda()
    label = torch.tensor(target, dtype=torch.long).view(1, -1).cuda()
    gid = torch.tensor(-1, dtype=torch.long).cuda()

    return Data(x=atom_feats, y=label, edge_index=bonds, edge_attr=bond_feats, gid=gid)


def get_atom_feats(crys, elem_feats):
    atoms = crys.atomic_numbers
    atom_feats = list()

    for i in range(0, len(atoms)):
        atom_feats.append(elem_feats[atoms[i] - 1, :])

    return numpy.vstack(atom_feats)


def get_bonds(crys, n_bond_feats, radius):
    rbf_means = numpy.linspace(start=1.0, stop=radius, num=n_bond_feats)
    list_nbrs = crys.get_all_neighbors(radius, include_index=True)
    bonds = list()
    bond_feats = list()

    for i in range(0, len(list_nbrs)):
        nbrs = list_nbrs[i]

        for j in range(0, len(nbrs)):
            bonds.append([i, nbrs[j][2]])
            bond_feats.append(rbf(numpy.full(n_bond_feats, nbrs[j][1]), rbf_means, beta=0.5))

    if len(bonds) == 0:
        return None, None

    return numpy.vstack(bonds), numpy.vstack(bond_feats)


def split_dataset(dataset, ratio):
    n_dataset1 = int(ratio * len(dataset))
    idx_rand = numpy.random.permutation(len(dataset))
    dataset1 = [dataset[idx] for idx in idx_rand[:n_dataset1]]
    dataset2 = [dataset[idx] for idx in idx_rand[n_dataset1:]]

    return dataset1, dataset2
