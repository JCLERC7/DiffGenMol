import deepsmiles
import deepchem as dc
import pandas as pd
import torch
from tqdm import tqdm
import utils

_, datasets, _ = dc.molnet.load_qm9(featurizer='ECFP')
train_smiles_prep = pd.DataFrame(data={'smiles': datasets[0].ids})

# Canonicalize smiles
train_smiles_prep['smiles'] = utils.canonicalize_smiles(train_smiles_prep['smiles'])
train_smiles = train_smiles_prep['smiles']

nb_smiles = 10

#print(train_smiles[:nb_smiles])
deepsmiles_converter = deepsmiles.Converter(rings=True, branches=True)
smiles, replace_dict, smiles_alphabet, largest_value_len = utils.preprocess_smiles(train_smiles[:nb_smiles], deepsmiles_converter)
featurizer = dc.feat.OneHotFeaturizer(charset=smiles_alphabet, max_length=largest_value_len)

continous_mol = utils.smiles_to_continous_mols(smiles, featurizer)
reconstitute_smiles = utils.canonicalize_smiles(utils.continous_mols_to_smiles(continous_mol, featurizer, replace_dict, deepsmiles_converter))

#print(reconstitute_smiles)

for i in range(len(train_smiles[:nb_smiles])):
    if train_smiles[i] != reconstitute_smiles[i]:
        print(train_smiles[i], '!=', reconstitute_smiles[i])
print('FIN')
print('test: avant')
print('O=C1[C@H]2C[N@H+](C2)[C@@]12CC2')
print('test: pendant')
print(deepsmiles_converter.encode('O=C1[C@H]2C[N@H+](C2)[C@@]12CC2'))
print('test: après')
print(utils.canonicalize_smiles([deepsmiles_converter.decode('O=C[C@H]C[N@H+]C4)[C@@]5CC3')])[0])
