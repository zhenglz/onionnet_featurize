from biopandas.mol2 import PandasMol2
from biopandas.mol2 import split_multimol2
from rdkit import Chem
from rdkit.Chem import AllChem
import subprocess as sp
import mdtraj as mt
import os
import numpy as np
import pandas as pd
from psikit import Psikit
import psi4


elements_ligand = ["H", "C", "O", "N", "P", "S", "HAX", "DU"]
elements_protein = ["H", "C", "O", "N", "P", "S", "HAX", "DU"]


def get_protein_elementtype(e):
    if e in elements_protein:
        return e
    else:
        return "DU"


def get_ligand_elementtype(e):
    '''if e == "C.ar":
        return "CAR"  '''
    if e.split(".")[0] in elements_ligand:
        return e.split(".")[0]
    else:
        return "DU"


class Molecule(object):
    """Small molecule parser object with Rdkit package.
    Parameters
    ----------
    in_format : str, default = 'smile'
        Input information (file) format.
        Options: smile, pdb, sdf, mol2, mol
    Attributes
    ----------
    molecule_ : rdkit.Chem.Molecule object
    mol_file : str
        The input file name or Smile string
    converter_ : dict, dict of rdkit.Chem.MolFrom** methods
        The file loading method dictionary. The keys are:
        pdb, sdf, mol2, mol, smile
    """

    def __init__(self, in_format="smile"):

        self.format = in_format
        self.molecule_ = None
        self.mol_file = None
        self.converter_ = None
        self.mol_converter()

    def mol_converter(self):
        """The converter methods are stored in a dictionary.
        Returns
        -------
        self : return an instance of itself
        """
        self.converter_ = {
            "pdb": Chem.MolFromPDBFile,
            "mol2": Chem.MolFromMol2File,
            "mol": Chem.MolFromMolFile,
            "smile": Chem.MolFromSmiles,
            "sdf": Chem.MolFromMolBlock,
            "pdbqt": self.babel_converter,
        }

        return self

    def babel_converter(self, mol_file, output):
        if os.path.exists(mol_file):
            try:
                cmd = 'obabel %s -O %s > /dev/null' % (mol_file, output)
                job = sp.Popen(cmd, shell=True)
                job.communicate()

                self.molecule_ = self.converter_['pdb']()
                return self.molecule_
            except:
                return None

    def load_molecule(self, mol_file):
        """Load a molecule to have a rdkit.Chem.Molecule object
        Parameters
        ----------
        mol_file : str
            The input file name or SMILE string
        Returns
        -------
        molecule : rdkit.Chem.Molecule object
            The molecule object
        """

        self.mol_file = mol_file
        if not os.path.exists(self.mol_file):
            print("Molecule file not exists. ")
            return None

        if self.format not in ["mol2", "mol", "pdb", "sdf", "pdbqt"]:
            print("File format is not correct. ")
            return None
        else:
            try:
                self.molecule_ = self.converter_[self.format](self.mol_file, sanitize=True,)
            except RuntimeError:
                return None

            return self.molecule_


class PdbqtParser(object):

    """Parse PDBQT file.

    Parameters
    ----------
    pdbqt_fn : str,
        Input pdbqt file.

    Examples
    --------
    >>> pdbqt = PdbqtParser("pdb.pdbqt")
    >>> df = pdbqt.to_dataframe()
    >>> df.values
    >>> df.columns
    >>> df['serial']

    """

    def __init__(self, pdbqt_fn=""):
        self.fn = pdbqt_fn

    def _get_atom_lines(self):
        if not os.path.exists(self.fn):
            print("INFO: No such pdbqt ")
            return []

        with open(self.fn) as lines:
            lines = [x for x in lines if x.startswith("ATOM")]

        return lines

    def _element_determinator(self, element, atomname):

        if len(element) == 1:
            if element == "A":
                return "CAR"
            elif element.upper() not in ['C', 'H', 'N', 'O', 'P', 'S', 'K', 'I', 'F']:
                print("INFO: find unusal element ", element, "and it will be replace by %s" % atomname[0].upper())
                return atomname[0].upper()
            else:
                return element.upper()
        elif len(element) == 2:
            e = element.upper()
            if e in ['BR', 'CL', 'MG', 'ZN']:
                return e
            else:
                return e[0]
        else:
            return "NA"

    def _parse_atom_line(self, line):
        _atom_id = int(line[6:11].strip())
        _atom_name = line[12:16].strip()
        _chainid = line[21]
        _res_name = line[17:20].strip()
        try:    
            _res_id = int(line[22:26].strip())
        except ValueError:
            _res_id = 0
        # coordination in unit: angstrom
        _x = float(line[30:38].strip())
        _y = float(line[38:46].strip())
        _z = float(line[46:54].strip())
        _element = self._element_determinator(line[76:79].strip(), _atom_name)
        try:
            _charge = float(line[70:76].strip())
        except ValueError:
            _charge = 0.0        

        return [_atom_id, _atom_name, _chainid, _res_name,
                _res_id, _x, _y, _z, _element, _charge]

    def to_dataframe(self):
        atom_lines = self._get_atom_lines()
        if atom_lines is None:
            return None
        else:
            _data = []
            for line in atom_lines:
                _data.append(self._parse_atom_line(line))

            _df = pd.DataFrame(_data, columns=['serial', 'name', 'chainID', 'resName',
                                               'resSeq', 'x', 'y', 'z', 'element', 'charge'])

            return _df


class ProteinParser(object):
    """Featurization of Protein-Ligand Complex based on
    onion-shape distance counts of atom-types.
    Parameters
    ----------
    pdb_fn : str
        The input pdb file name. The file must be in PDB format.
    Attributes
    ----------
    pdb : mdtraj.Trajectory
        The mdtraj.trajectory object containing the pdb.
    receptor_indices : np.ndarray
        The receptor (protein) atom indices in mdtraj.Trajectory
    rec_ele : np.ndarray
        The element types of each of the atoms in the receptor
    pdb_parsed_ : bool
        Whether the pdb file has been parsed.
    distance_computed : bool
        Whether the distances between atoms in receptor and ligand has been computed.
    Examples
    --------
    >>> pdb = ProteinParser("input.pdb")
    >>> pdb.parsePDB('protein and chainid 0')
    >>> pdb.coordinates_
    >>> print(pdb.rec_ele)
    """

    def __init__(self, pdb_fn):
        self.pdb = mt.load_pdb(pdb_fn)

        self.receptor_indices = np.array([])
        self.rec_ele = np.array([])

        self.pdb_parsed_ = False
        self.coordinates_ = None

    def get_coordinates(self):
        """
        Get the coordinates in the pdb file given the receptor indices.
        Returns
        -------
        self : an instance of itself
        """
        # unit: angstrom
        self.coordinates_ = self.pdb.xyz[0][self.receptor_indices] * 10.0 # bug fixed for atomic unit

        return self

    def parsePDB(self, rec_sele="protein"):
        """
        Parse the pdb file and get the detail information of the protein.
        Parameters
        ----------
        rec_sele : str,
            The string for protein selection. Please refer to the following link.
        References
        ----------
        Mdtraj atom selection language: http://mdtraj.org/development/atom_selection.html
        Returns
        -------
        """
        top = self.pdb.topology
        # obtain the atom indices of the protein
        self.receptor_indices = top.select(rec_sele)
        _table, _bond = top.to_dataframe()

        # fetch the element type of each one of the protein atom
        self.rec_ele = _table['element'][self.receptor_indices].values
        # fetch the coordinates of each one of the protein atom
        self.get_coordinates()

        self.pdb_parsed_ = True

        return self


class LigandParser(object):
    """Parse the ligand with biopanda to obtain coordinates and elements.
    Parameters
    ----------
    ligand_fn : str,
        The input ligand file name.
    Methods
    -------
    Attributes
    ----------
    lig : a biopandas mol2 read object
    lig_data : a panda data object holding the atom information
    coordinates : np.ndarray, shape = [ N, 3]
        The coordinates of the atoms in the ligand, N is the number of atoms.
    """

    def __init__(self, ligand_fn):
        self.lig_file = ligand_fn
        self.lig = None
        self.lig_data = None

        self.lig_ele = None
        self.coordinates_ = None
        self.mol2_parsed_ = False

    def _format_convert(self, input, output):
        mol = Molecule(in_format=input.split(".")[-1])
        mol.babel_converter(input, output)
        return self

    def get_element(self):
        ele = list(self.lig_data["atom_type"].values)
        self.lig_ele = list(map(get_ligand_elementtype, ele))
        return self

    def get_coordinates(self):
        """
        Get the coordinates in the pdb file given the ligand indices.
        Returns
        -------
        self : an instance of itself
        """
        self.coordinates_ = self.lig_data[['x', 'y', 'z']].values
        return self

    def parseMol2(self):
        if not self.mol2_parsed_:
            if self.lig_file.split(".")[-1] != "mol2":
                out_file = self.lig_file + ".mol2"
                self._format_convert(self.lig_file, out_file)
                self.lig_file = out_file

            if os.path.exists(self.lig_file):
                try:
                    self.lig = PandasMol2().read_mol2(self.lig_file)
                except ValueError:
                    templ_ligfile = self.lig_file + "templ.pdb"
                    self._format_convert(self.lig_file, templ_ligfile)
                    if os.path.exists(templ_ligfile):
                        self.lig = mt.load_pdb(templ_ligfile)
                        top = self.lig.topology
                        table, bond = top.to_dataframe()
                        self.lig_ele = list(table['element'])
                        # nano-meter to angstrom
                        self.coordinates_ = self.lig.xyz[0] * 10.0
                        self.lig_data = table
                        self.lig_data['x'] = self.coordinates_[:, 0]
                        self.lig_data['y'] = self.coordinates_[:, 1]
                        self.lig_data['z'] = self.coordinates_[:, 2]
                        self.mol2_parsed_ = True
                        os.remove(templ_ligfile)
                        return self
            else:
                return None

            self.lig_data = self.lig.df
            self.get_element()
            self.get_coordinates()
            self.mol2_parsed_ = True

        return self


class CompoundBuilder(object):
    """Generate 3D coordinates of compounds.
    Parameters
    ----------
    in_format : str, default='smile'
        The input file format. Options are
        pdb, sdf, mol2, mol and smile.
    out_format : str, default='pdb'
        The output file format. Options are
        pdb, sdf, mol2, mol and smile.
    addH : bool, default = True
        Whether add hydrogen atoms
    optimize : bool, default = True
        Whether optimize the output compound conformer
    Attributes
    ----------
    mol_file : str
        The input file name or smile string.
    molecule_ : rdkit.Chem.Molecule object
        The target compound molecule object
    add_H : bool, default = True
    Examples
    --------
    >>> # generate 3D conformation from a SMILE code and save as a pdb file
    >>> from drugmap import builder
    >>> comp = builder.CompoundBuilder(out_format="pdb", in_format="smile")
    >>> comp.in_format
    'smile'
    >>> comp.load_mol("CCCC")
    <deepunion.builder.CompoundBuilder object at 0x7f0cd1d909b0>
    >>> comp.generate_conformer()
    <deepunion.builder.CompoundBuilder object at 0x7f0cd1d909b0>
    >>> comp.write_mol("mol_CCCC.pdb")
    <deepunion.builder.CompoundBuilder object at 0x7f0cd1d909b0>
    >>> # the molecule has been saved to mol_CCCC.pdb in working directory
    >>> # convert the pdb file into a pdbqt file
    >>> babel_converter("mol_CCCC.pdb", "mol_CCCC.pdbqt")
    """
    def __init__(self, out_format="pdb", in_format="smile",
                 addHs=True, optimize=True):
        self.out_format = out_format
        self.mol_file = None
        self.molecule_ = None
        self.in_format = in_format
        self.add_H = addHs
        self.optimize_ = optimize
        self.optimize_status_ = False
        self.converter_ = None
        self.write_converter()
    def generate_conformer(self):
        """Generate 3D conformer for the molecule.
        The hydrogen atoms are added if necessary.
        And the conformer is optimized with a rdkit MMFF
        optimizer
        Returns
        -------
        self : return an instance of itself
        References
        ----------
        Halgren, T. A. “Merck molecular force field. I. Basis, form,
        scope, parameterization, and performance of MMFF94.” J. Comp.
        Chem. 17:490–19 (1996).
        https://www.rdkit.org/docs/GettingStartedInPython.html
        """
        if self.molecule_ is not None:
            # add hydrogen atoms
            if self.add_H:
                self.molecule_ = Chem.AddHs(self.molecule_)
            # generate 3D structure
            AllChem.EmbedMolecule(self.molecule_)
            # optimize molecule structure
            if self.optimize_:
                try:
                    AllChem.MMFFOptimizeMolecule(self.molecule_)
                    self.optimize_status_ = True
                except ValueError:
                    pass
            return self
        else:
            print("Load molecule first. ")
            return self
    def load_mol(self, mol_file):
        """Load a molecule from a file or a SMILE string
        Parameters
        ----------
        mol_file : str
            The input molecule file name or a SMILE string
        Returns
        -------
        self : return an instance of itself
        """
        self.mol_file = mol_file
        mol = Molecule(in_format=self.in_format)
        self.molecule_ = mol.load_molecule(mol_file)
        return self

    def write_converter(self):
        """Write file methods.
        Returns
        -------
        self : an instance of itself
        """
        converter = {
            "pdb": Chem.MolToPDBFile,
            "sdf": Chem.MolToMolBlock,
            #"mol2": Chem.MolToMol2File,
            "mol": Chem.MolToMolFile,
            "smile": Chem.MolToSmiles,
        }
        self.converter_ = converter
        return self

    def write_mol(self, out_file="compound.pdb"):
        """Write a molecule to a file.
        Parameters
        ----------
        out_file : str, default = compound.pdb
            The output file name.
        Returns
        -------
        self : an instance of itself
        """
        # need to load file first
        self.generate_conformer()
        self.converter_[self.out_format](self.molecule_, out_file)
        return self


class OptimizerQM(object):
    def __init__(self, mol_file, in_format="smile"):
        self.molfn = mol_file
        self.format_ = in_format
        self.molecule_ = None
        self.psikit_obj_ = None
        self.optimize_status_ = False

    def optimize(self, basis_set='hf/6-31g**', maxiter=1000):
        if self.format_ == "smile":
            pk = Psikit()
            pk.read_from_smiles(self.molfn)
            try:
                pk.optimize(basis_set, maxiter=maxiter)
                self.molecule_ = pk.mol
                self.psikit_obj_ = pk
                self.optimize_status_ = True
            except psi4.OptimizationConvergenceError:
                cb = CompoundBuilder()
                cb.load_mol(self.molfn)
                cb.generate_conformer()
                self.optimize_status_ = cb.optimize_status_
                self.molecule_ = cb.optimize_status_
                self.psikit_obj_ = None
            return self
        else:
            return None


def babel_converter(input, output, babelexe="obabel", mode="general"):
    cmd = ""
    if mode == "general":
        cmd = "%s %s -O %s" % (babelexe, input, output)
    elif mode == "AddPolarH":
        cmd = "%s %s -O %s -d" % (babelexe, input, "xxx_temp_noH.pdbqt")
        job = sp.Popen(cmd, shell=True)
        job.communicate()
        cmd = "%s %s -O %s --AddPolarH" % (babelexe, "xxx_temp_noH.pdbqt", output)
    else:
        pass
    job = sp.Popen(cmd, shell=True)
    job.communicate()
    return None
