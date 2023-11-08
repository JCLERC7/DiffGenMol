import deepchem as dc
import torch
smiles = ['CCC','CCC','Cn1cnc2c1c(=O)n(C)c(=O)n2C','CC(=O)N1CN(C(C)=O)[C@@H](O)[C@@H]1O']
largest_smiles_len = len(max(smiles, key=len))

featurizer = dc.feat.OneHotFeaturizer(max_length=largest_smiles_len)

encodings = featurizer.featurize(smiles)
print(type(encodings[0]))
print(encodings[0].shape)
print(encodings[0][0])
print(featurizer.untransform(encodings[0]))

len_selfies_alphabet = encodings[0].shape[1]

print(largest_smiles_len)

onehots = torch.zeros(len(smiles), largest_smiles_len, len_selfies_alphabet, dtype=torch.float)
for i, encoding in enumerate(encodings):
    onehots[i, :, :] = torch.tensor(encoding, dtype=torch.float32)
input_tensor = onehots.view(len(encodings), -1)
dequantized_onehots = input_tensor.add(torch.rand(input_tensor.shape, dtype=torch.float32))
continous_mols = dequantized_onehots.div(2)
#print(continous_mols)

denormalized_data = continous_mols * 2
quantized_data = torch.floor(denormalized_data)
quantized_data = torch.clip(quantized_data, 0, 1)
mols_list = quantized_data.cpu().int().numpy().tolist()

for mol in mols_list:
    for i in range(largest_smiles_len):
        row = mol[len_selfies_alphabet * i: len_selfies_alphabet * (i + 1)]
        if all(elem == 0 for elem in row):
            mol[len_selfies_alphabet * (i+1) - 1] = 1

mols = []
for mol in torch.tensor(mols_list, dtype=torch.float32).view(len(encodings), largest_smiles_len, len_selfies_alphabet):
    mols.append(featurizer.untransform(mol))
print(mols)