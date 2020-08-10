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


def atomic_distance(dat):
    return np.sqrt(np.sum(np.square(dat[0] - dat[1])))


def distance_pairs(coord_pro, coord_lig):
    pairs = list(itertools.product(coord_pro, coord_lig))
    distances = map(atomic_distance, pairs)

    return np.array(list(distances))


def distance_pairs_mdtraj(coord_pro, coord_lig):
    xyz = np.concatenate((coord_pro, coord_lig), axis=0)
    #print(xyz.shape)
    # mdtraj use nanometer for coordinations,
    # convert angstrom to nanometer
    xyz = xyz.reshape((1, -1, 3)) * 0.1
    # for the xyz, the sample is (N_frames, N_atoms, N_dim)
    # N_frames, it is usually 1 for a normal single-molecule PDB
    # N_atoms is the number of atoms in the pdb file
    # N_dims is the dimension of coordinates, 3 here for x, y and z

    # create a mdtraj Trajectory object,Topology object could be ignored.
    traj = mt.Trajectory(xyz=xyz, topology=None)
    # create a list of atom-pairs from atom index of protein and ligand
    atom_pairs = itertools.product(np.arange(coord_pro.shape[0]),
                                   np.arange(coord_pro.shape[0], coord_pro.shape[0]+coord_lig.shape[0]))

    # convert the distance to angstrom from nanometer.
    # Actually we could just leave it as angstrom from the beginning for faster calculation,
    # but it is more reasonable to do it in order to aligning with mdtraj-style calculation.
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


def complex_featurizer(pro_fn, lig_fn, n_cutoffs, output, mode="mdtraj"):
    print("INFO: Processing %s and %s ... with mode %s" % (pro_fn, lig_fn, mode))

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
                    # determine to use with distance calculation engine
                    if mode.lower() == "numpy":
                        distances = fast_distance_pairs(protein_xyz, ligand_xyz)
                    else:
                        distances = distance_pairs_mdtraj(protein_xyz, ligand_xyz)

                    # iterative over all the distance cutoffs
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

        # save the results for one-ligand-code per line/row in the result list
        results.append(list(onionnet_counts.values.ravel()))
        results_code.append(mol2_code)

        os.remove(fn)

    df = pd.DataFrame(results, columns=_columns)
    df.index = results_code

    df.to_csv(output, header=True, index=True)

