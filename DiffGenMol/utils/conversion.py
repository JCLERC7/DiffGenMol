import numpy as np
import selfies as sf
import torch
from rdkit import Chem

# Helper
def keys_int(symbol_to_int):
  d={}
  i=0
  for key in symbol_to_int.keys():
    d[i]=key
    i+=1
  return d

def smiles_to_selfies(smiles):
   sf.set_semantic_constraints()  # reset constraints
   constraints = sf.get_semantic_constraints()
   constraints['?'] = 5
   constraints['Se'] = 6
   constraints['P-1'] = 6
   constraints['I'] = 5
   sf.set_semantic_constraints(constraints)
   selfies_list = np.asanyarray(smiles.apply(sf.encoder))
   return selfies_list

def get_selfies_features(selfies):
  selfies_alphabet = sf.get_alphabet_from_selfies(selfies)
  selfies_alphabet.add('[nop]')  # Add the "no operation" symbol as a padding character
  selfies_alphabet.add('.') 
  selfies_alphabet = list(sorted(selfies_alphabet))
  largest_selfie_len = max(sf.len_selfies(s) for s in selfies)
  symbol_to_int = dict((c, i) for i, c in enumerate(selfies_alphabet))
  int_mol=keys_int(symbol_to_int)
  return largest_selfie_len, selfies_alphabet, symbol_to_int, int_mol

def selfies_to_continous_mols(selfies, largest_selfie_len, selfies_alphabet, symbol_to_int):
   onehots = torch.zeros(len(selfies), largest_selfie_len, len(selfies_alphabet), dtype=torch.float)
   for i, selfie in enumerate(selfies):
    one_hot = sf.selfies_to_encoding(selfie, symbol_to_int, largest_selfie_len, enc_type='one_hot')
    onehots[i, :, :] = torch.tensor(one_hot, dtype=torch.float32)
   input_tensor = onehots.view(len(selfies), -1)
   dequantized_onehots = input_tensor.add(torch.rand(input_tensor.shape, dtype=torch.float32))
   continous_mols = dequantized_onehots.div(2)
   return continous_mols

def continous_mols_to_selfies(continous_mols, selfies_alphabet, largest_selfie_len, int_mol):
   denormalized_data = continous_mols * 2
   quantized_data = torch.floor(denormalized_data)
   quantized_data = torch.clip(quantized_data, 0, 1)
   mols_list = quantized_data.cpu().int().numpy().tolist()
   for mol in mols_list:
    for i in range(largest_selfie_len):
        row = mol[len(selfies_alphabet) * i: len(selfies_alphabet) * (i + 1)]
        if all(elem == 0 for elem in row):
            mol[len(selfies_alphabet) * (i+1) - 1] = 1
   selfies = sf.batch_flat_hot_to_selfies(mols_list, int_mol)
   return selfies

def selfies_to_mols(selfies_to_convert):
  valid_count = 0
  valid_selfies, invalid_selfies = [], []
  mols = []
  for idx, selfies in enumerate(selfies_to_convert):
    try:
      if Chem.MolFromSmiles(sf.decoder(selfies_to_convert[idx]), sanitize=True) is not None:
          valid_count += 1
          valid_selfies.append(selfies)
          mols.append(Chem.MolFromSmiles(sf.decoder(selfies_to_convert[idx])))
      else:
        invalid_selfies.append(selfies)
    except Exception:
      pass
  return mols, valid_selfies, valid_count

def mols_to_smiles(mols):
   smiles = [Chem.MolToSmiles(mol) for mol in mols]
   return smiles

def discretize_continuous_values(values, num_classes, breakpoints = None):
  if breakpoints is None:
    sorted_values = sorted(values)
    breakpoints = [sorted_values[i * len(values) // num_classes] for i in range(1, num_classes)]
  discretized_values = []
  for value in values:
      class_value = sum(value > breakpoint for breakpoint in breakpoints)
      discretized_values.append(class_value)
  return discretized_values, breakpoints

