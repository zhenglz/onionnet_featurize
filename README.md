
# USAGE

    cd examples/multiple_molecules_mol2_CASF
    # calculating distance with mdtraj 
    python ../../gen_features.py -inp input.dat -mode mdtraj

    # calculating distance with numpy
    python ../../gen_features.py -inp input.dat -mode numpy

# DATASETS

Including training, validating and testing features, as well corresponding labels.
    
    cd ./datasets
    
# Installation

    conda install -c anaconda numpy pandas
    conda install -c omnia mdtraj
    conda install -c conda-forge biopandas
    conda install -c rdkit rdkit 
    
    # optional 
    conda install -c psi4
    conda install -c iwatobipen psikit
