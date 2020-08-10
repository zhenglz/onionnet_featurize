
# Featurization of PDBbind v2018 based on OnionNet version 1
## Command used:
  
    python gen_features.py -inp input_testv2016_onnetv1

    # then for each target (one pdb code), a csv output file 
    # will be generated in its corresponding folder as stated in
    # input_testv2016_onnetv1 column #3

## All related features pre-generated: onnetv1_features.tgz
Untar the file, and you will have 3 csv files for train (11883), validate(996) and test (290). Related labels could also be found in this folder. 
Please be careful, the labels for training contains only 11881 samples, there are two mismatches with the train feature file. This will be fixed soon.

