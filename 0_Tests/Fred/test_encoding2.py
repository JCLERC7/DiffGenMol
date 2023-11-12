import deepchem as dc
import utils
import torch
smiles = ['CCC','CCC','Cn1cnc2c1c(=O)n(C)c(=O)n2C','CC(=O)N1CN(C(C)=O)[C@@H](O)[C@@H]1O']
largest_smiles_len = len(max(smiles, key=len))

featurizer = dc.feat.OneHotFeaturizer(max_length=largest_smiles_len)

encodings = featurizer.featurize(smiles)

print('encodings')
print(encodings.shape)

print(type(encodings[0]))
print(encodings[0].shape)
print(encodings[0][0])
print(encodings[0][1])
print(encodings[0][2])
print(encodings[0][3])
print(encodings[0][4])
print(featurizer.untransform(encodings[0]))

len_selfies_alphabet = encodings[0].shape[1]

print(largest_smiles_len)

#onehots = torch.zeros(len(smiles), largest_smiles_len, len_selfies_alphabet, dtype=torch.float)
#for i, encoding in enumerate(encodings):
#    onehots[i, :, :] = torch.tensor(encoding, dtype=torch.float32)
#input_tensor = onehots.view(len(encodings), -1)
#dequantized_onehots = input_tensor.add(torch.rand(input_tensor.shape, dtype=torch.float32))
onehots = torch.tensor(encodings, dtype=torch.float32)
dequantized_onehots = onehots.add(torch.rand(onehots.shape, dtype=torch.float32))
continous_mols = dequantized_onehots.div(2)
print('continous_mols:')
print(continous_mols.shape)

denormalized_data = continous_mols * 2
quantized_data = torch.floor(denormalized_data)
quantized_data = torch.clip(quantized_data, 0, 1)
for mol in quantized_data:
  for letter in mol:
    if all(elem == 0 for elem in letter):
      letter[len_selfies_alphabet] = 1
smiles = [featurizer.untransform(mol.cpu().tolist()) for mol in quantized_data]

print(smiles)

smiles = ['CCC','CCC','Cn1cnc2c1c(=O)n(C)c(=O)n2C','CC(=O)N1CN(C(C)=O)[C@@H](O)[C@@H]1O']

largest_smiles_len, smiles_alphabet = utils.preprocess_smiles(smiles)
featurizer = dc.feat.OneHotFeaturizer(charset=smiles_alphabet, max_length=largest_smiles_len)

print('test')
print(utils.one_smiles_to_continous_mol(smiles[0], featurizer).shape)
print('test2')
print(utils.continous_mols_to_smiles(utils.one_smiles_to_continous_mol(smiles[0], featurizer)[None, :], featurizer))


encodings = featurizer.featurize([smiles[0]])
print('encodings.shapeINDIC')
print(encodings.shape)
print(featurizer.untransform(encodings.squeeze()))

print(utils.continous_mols_to_smiles(utils.smiles_to_continous_mols(smiles,featurizer),featurizer))

continous_mols = utils.one_smiles_to_continous_mol(smiles[3],featurizer)[None, :]

print(continous_mols.shape)

print(utils.continous_mols_to_smiles(continous_mols,featurizer))
