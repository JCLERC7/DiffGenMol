import pandas as pd
import selfies as sf
import torch
import utils

# Smiles to selfies to continous one-hot 
smiles = ['CCC', 'CC1NC11C(C)OC1=N', 'CC(C)NC1=NNC=C1']
smiles = utils.canonicalize_smiles(smiles)
print('Input:')
print(smiles)

df_smiles = pd.DataFrame(smiles, columns=['smiles'])

selfies = utils.smiles_to_selfies(df_smiles['smiles'])
largest_selfie_len, selfies_alphabet, symbol_to_int, int_mol = utils.get_selfies_features(selfies)
print(selfies_alphabet)

onehots = torch.zeros(len(selfies), largest_selfie_len, len(selfies_alphabet), dtype=torch.float)
for i, selfie in enumerate(selfies):
    one_hot = sf.selfies_to_encoding(selfie, symbol_to_int, largest_selfie_len, enc_type='one_hot')
    onehots[i, :, :] = torch.tensor(one_hot, dtype=torch.float32)

continous_mols = utils.selfies_to_continous_mols(selfies, largest_selfie_len, selfies_alphabet, symbol_to_int)
continous_mol = utils.one_selfies_to_continous_mol(selfies[1], largest_selfie_len, symbol_to_int)

# Continous one-hot to selfies to smiles
output_selfies = utils.continous_mols_to_selfies(continous_mols, selfies_alphabet, int_mol)
output_mols, _, _ = utils.selfies_to_mols(output_selfies)
output_smiles = utils.mols_to_smiles(output_mols)
output_smiles = utils.canonicalize_smiles(output_smiles)
print('output:')
print(output_smiles)

output_selfies = utils.continous_mols_to_selfies(continous_mol[None, :], selfies_alphabet, int_mol)
output_mols, _, _ = utils.selfies_to_mols(output_selfies)
output_smiles = utils.mols_to_smiles(output_mols)
output_smiles = utils.canonicalize_smiles(output_smiles)
print('output:')
print(output_smiles)
