import multiprocessing
import sys, os
import argparse
from argparse import RawDescriptionHelpFormatter
import numpy as np
from utils.contact_featurizer import complex_featurizer


def featurize(args):
    pfn, lfn, output, n_cutoffs, mode = args

    if not os.path.exists(pfn) or not os.path.exists(lfn):
        return None
    #print("Now calculating ", pfn, lfn)
    complex_featurizer(pfn, lfn, n_cutoffs, output, mode=mode)

    return None


def main():
    d = """
       Predicting protein-ligand binding affinities (pKa) with OnionNet model.

       Citation: Zheng L, Fan J, Mu Y. arXiv preprint arXiv:1906.02418, 2019.
       Author: Liangzhen Zheng (zhenglz@outlook.com)
       This script is used to generate inter-molecular element-type specific
       contact features. Installation instructions should be refered to
       https://github.com/zhenglz/onionnet-v2

       Examples:
       Show help information
       python generate_features.py -h
       
       Run the script
       python generate_features.py -inp input_samples.dat -out features_samples.csv
       # tutorial example
       
       cd examples/single_molecule_mol2/10gs
       python ../../generate_features.py -inp input.dat -mode \"mdtraj\" 
       
    """
    parser = argparse.ArgumentParser(description=d,
                                     formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-inp", type=str, default="input.dat",
                        help="Input. The input file containg the file path of each \n"
                             "of the protein-ligand complexes files (in pdb format.)\n"
                             "There should be only 2 columns, each row or line containing\n"
                             "the input file path, relative or absolute path to protein and ligand respectively.")
    #parser.add_argument("-fdpath", type=str, default="./", help="input pdb files ")
    #parser.add_argument("-out", default="features/", type=str, help="output file prefix")
    parser.add_argument("-nt", type=int, default=1,
                        help="Input, optional. Default is 1. "
                             "Use how many of cpu cores.")
    parser.add_argument("-upbound", type=float, default=31.0,
                        help="Input, optional. Default is 31 angstrom. "
                             "The largest distance cutoff.")
    parser.add_argument("-lowbound", type=float, default=1.0,
                        help="Input, optional. Default is 1.0 angstrom. "
                             "The lowest distance cutoff.")
    parser.add_argument("-nbins", type=int, default=60,
                        help="Input, optional. Default is 60. "
                             "The number of distance cutoffs.")
    parser.add_argument("-use_columbic", type=int, default=0,
                        help="Calculate columbic based contacts.")
    parser.add_argument("-mode", default="mdtraj", type=str,
                        help="Usage with mode for calculation. "
                             "numpy: numpy implemention for distance. \n"
                             "mdtraj: mdtraj engine for distance calculation.")

    args = parser.parse_args()

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    USE_COLUMBIC = args.use_columbic
    print("INFO: Featurization with mode = ", args.mode)

    num_threads = args.nt
    n_cutoffs = np.linspace(args.lowbound,
                            args.upbound,
                            args.nbins)

    with open(args.inp) as lines:
        file_paths = [x.split() for x in lines if
                     ("#" not in x and len(x.split()))]

    input_arguments = []
    for item in file_paths:
        input_arguments.append(item +[n_cutoffs, args.mode])

    if num_threads <= 1:
        for args in input_arguments:
            featurize(args)
    else:
        pool = multiprocessing.Pool(num_threads)
        pool.map(featurize, input_arguments)

        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
