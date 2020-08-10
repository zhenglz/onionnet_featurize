#!/usr/bin/env python

import numpy as np
import pandas as pd
from biopandas.mol2 import split_multimol2
import itertools
import os
import mdtraj as mt
from .utils import LigandParser, ProteinParser


# Define required atomic element type
#elements_ligand = ["H", "C", "CAR", "O", "N",
#                   "S", "P", "DU", "Br", "Cl", "F"]
#elements_protein = ["H", "C", "O", "N", "S", "DU"]
elements_ligand = ["H", "C", "O", "N", "P", "S", "HAX", "DU"]
elements_protein = ["H", "C", "O", "N", "P", "S", "HAX", "DU"]

'''
def split_multimol2(mol2_path):
    r"""
    Splits a multi-mol2 file into individual Mol2 file contents.

    Parameters
    -----------
    mol2_path : str
      Path to the multi-mol2 file. Parses gzip files if the filepath
      ends on .gz.

    Returns
    -----------
    A generator object for lists for every extracted mol2-file. Lists contain
        the molecule ID and the mol2 file contents.
        e.g., ['ID1234', ['@<TRIPOS>MOLECULE\n', '...']]. Note that bytestrings
        are returned (for reasons of efficieny) if the Mol2 content is read
        from a gzip (.gz) file.

    """

    open_file = open
    read_mode = 'r'
    check = {'rb': b'@<TRIPOS>MOLECULE', 'r': '@<TRIPOS>MOLECULE'}

    mol2_data = []

    with open_file(mol2_path, read_mode) as f:
        mol2 = ['', []]
        while True:
            try:
                line = next(f)
                if line.startswith(check[read_mode]):
                    if mol2[0]:
                        mol2_data.append(mol2)
                    mol2 = ['', []]
                    mol2_id = next(f)
                    mol2[0] = mol2_id.rstrip()
                    mol2[1].append(line)
                    mol2[1].append(mol2_id)
                else:
                    mol2[1].append(line)
            except StopIteration:
                mol2_data.append(mol2)
                return mol2_data
'''

def atomic_distance(dat):
    return np.sqrt(np.sum(np.square(dat[0] - dat[1])))


def distance_pairs(coord_pro, coord_lig):
    pairs = list(itertools.product(coord_pro, coord_lig))
    distances = map(atomic_distance, pairs)

    return np.array(list(distances))


def distance_pairs_mdtraj(coord_pro, coord_lig):
    xyz = np.concatenate((coord_pro, coord_lig), axis=0)
    #print(xyz.shape)
    xyz = xyz.reshape((1, -1, 3)) * 0.1

    traj = mt.Trajectory(xyz=xyz, topology=None)
    #print("MDTRAJ XYZ shape ", traj.xyz.shape)
    #print("Protein index", np.arange(coord_pro.shape[0]).shape)
    #print("Protein index", np.arange(coord_pro.shape[0]).shape)

    atom_pairs = itertools.product(np.arange(coord_pro.shape[0]),
                                   np.arange(coord_pro.shape[0], coord_pro.shape[0]+coord_lig.shape[0]))

    dist = mt.compute_distances(traj, atom_pairs)[0] * 10.0

    return dist


def fast_distance_pairs(coords_pro, coords_lig):
    return np.array([np.linalg.norm(coord_pro - coords_lig, axis=-1) for coord_pro in coords_pro]).ravel()


def distance2counts(megadata):
    d, c, charge_pairs, distance_mode, distance_delta = megadata

    if charge_pairs is None:
        return np.sum((np.array(d) <= c) * 1.0)
    else:
        atompair_in_dist_range = ((np.array(d) <= c) & (np.array(d) < c - distance_delta)) * 1.0

        if distance_mode == 'cutoff':
            return np.multiply(atompair_in_dist_range, np.array(charge_pairs) / c)
        else:
            return np.multiply(atompair_in_dist_range, np.divide(charge_pairs, d))


def complex_featurizer(pro_fn, lig_fn, n_cutoffs, output, mode="numpy"):
    print("INFO: Processing %s and %s ..." % (pro_fn, lig_fn))

    # get protein elements and protein xyz
    pro = ProteinParser(pro_fn)
    pro.parsePDB()
    protein_data = pd.DataFrame()
    protein_data["element"] = pro.rec_ele
    # print(pro.rec_ele)
    for i, d in enumerate(['x', 'y', 'z']):
        # coordinates by mdtraj in unit nanometer
        protein_data[d] = pro.coordinates_[:, i]

    results = []
    results_code = []

    # processing multiple-molecule mol2 file
    for counter, mol2 in enumerate(split_multimol2(lig_fn)):
        mol2_code, mol2_lines = mol2
        print("INFO: #%d Ligand ID code " % counter, mol2_code)

        fn = str(pro_fn) + "_" + str(mol2_code) + ".mol2"
        with open(fn, 'w') as tofile:
            for line in mol2_lines:
                tofile.write(line)
        tofile.close()

        # get ligand mol2 elements and coordinates xyz
        lig = LigandParser(fn)
        lig.parseMol2()
        ligand_data = pd.DataFrame()
        ligand_data['element'] = lig.lig_ele
        for i, d in enumerate(['x', 'y', 'z']):
            # the coordinates in ligand are in angstrom
            ligand_data[d] = lig.coordinates_[:, i]

        onionnet_counts = pd.DataFrame()

        for ep in elements_protein:
            for el in elements_ligand:
                # for each element combination, such as C-O, we need to evaluate the contacts
                # at 60 different distance cutoff

                protein_xyz = protein_data[protein_data['element'] == ep][['x', 'y', 'z']].values
                ligand_xyz  = ligand_data[ligand_data['element'] == el][['x', 'y', 'z']].values

                counts       = np.zeros(len(n_cutoffs))
                total_counts = 0 

                if protein_xyz.shape[0] and ligand_xyz.shape[0]:
                    if mode.lower() == "numpy":
                        distances = fast_distance_pairs(protein_xyz, ligand_xyz)
                    else:
                        distances = distance_pairs_mdtraj(protein_xyz, ligand_xyz)
                    
                    for i, c in enumerate(n_cutoffs):
                        single_count = distance2counts((distances, c, None, '', 0.0))
                        # get the contacts in each shell by substract all contacts in previous shells
                        single_count  = single_count - total_counts
                        counts[i]     = single_count
                        total_counts += single_count

                feature_id = "%s_%s" % (el, ep)
                onionnet_counts[feature_id] = counts

        _columns = list(onionnet_counts.columns) * n_cutoffs.shape[0]
        for i in range(len(_columns)):
            _columns[i] = _columns[i] + "_" + str(i)

        # we save the results for one-ligand per line/row in the dataframe
        results.append(list(onionnet_counts.values.ravel()))
        results_code.append(mol2_code)

        os.remove(fn)

    df = pd.DataFrame(results, columns=_columns)
    df.index = results_code

    df.to_csv(output, header=True, index=True)

