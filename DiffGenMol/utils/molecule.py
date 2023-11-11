import deepchem as dc
from guacamol.utils.chemistry import canonicalize
import numpy as np
import os
import selfies as sf
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import RDConfig
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

# Helper
def keys_int(symbol_to_int):
  d={}
  i=0
  for key in symbol_to_int.keys():
    d[i]=key
    i+=1
  return d

def smiles_to_selfies(smiles):
   selfies_list = np.asanyarray(smiles.apply(sf.encoder))
   return selfies_list

def get_selfies_features(selfies):
  selfies_alphabet = sf.get_alphabet_from_selfies(selfies)
  if '[/F]' not in selfies_alphabet:
    selfies_alphabet.add('[/F]')
  if '[\\CH1-1]' not in selfies_alphabet:
    selfies_alphabet.add('[\\CH1-1]')
  selfies_alphabet.add('[nop]')  # Add the "no operation" symbol as a padding character
  selfies_alphabet.add('.') 
  selfies_alphabet = list(sorted(selfies_alphabet))
  largest_selfie_len = max(sf.len_selfies(s) for s in selfies)
  symbol_to_int = dict((c, i) for i, c in enumerate(selfies_alphabet))
  int_mol=keys_int(symbol_to_int)
  return largest_selfie_len, selfies_alphabet, symbol_to_int, int_mol

def get_smiles_features(smiles):
  smiles_alphabet = ['#', ')', '(', '+', '-', '/', '1', '3', '2', '5', '4', '7', '6', '8', 
                     '=', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'S', '[', ']', '\\', 'c',
                       'l', 'o', 'n', 'p', 's', 'r', 'Br', 'Cl', 'Cn', 'Sc', 'Se']
  largest_smiles_len = len(max(smiles, key=len))
  return largest_smiles_len, smiles_alphabet

def selfies_to_continous_mols(selfies, largest_selfie_len, selfies_alphabet, symbol_to_int):
   onehots = torch.zeros(len(selfies), largest_selfie_len, len(selfies_alphabet), dtype=torch.float)
   for i, selfie in enumerate(selfies):
    one_hot = sf.selfies_to_encoding(selfie, symbol_to_int, largest_selfie_len, enc_type='one_hot')
    onehots[i, :, :] = torch.tensor(one_hot, dtype=torch.float32)
   dequantized_onehots = onehots.add(torch.rand(onehots.shape, dtype=torch.float32))
   continous_mols = dequantized_onehots.div(2)
   return continous_mols

def smiles_to_continous_mols(smiles, featurizer):
   encodings = featurizer.featurize(smiles)
   onehots = torch.tensor(encodings, dtype=torch.float32)
   dequantized_onehots = onehots.add(torch.rand(onehots.shape, dtype=torch.float32))
   continous_mols = dequantized_onehots.div(2)
   return continous_mols

def one_selfies_to_continous_mol(selfies, largest_selfie_len, symbol_to_int):
   one_hot = sf.selfies_to_encoding(selfies, symbol_to_int, largest_selfie_len, enc_type='one_hot')
   onehot = torch.tensor(one_hot, dtype=torch.float32)
   dequantized_onehot = onehot.add(torch.rand(onehot.shape, dtype=torch.float32))
   continous_mol = dequantized_onehot.div(2)
   return continous_mol

def one_smiles_to_continous_mol(smiles, featurizer):
  encodings = featurizer.featurize([smiles])
  onehot = torch.tensor(encodings.squeeze(), dtype=torch.float32)
  dequantized_onehot = onehot.add(torch.rand(onehot.shape, dtype=torch.float32))
  continous_mol = dequantized_onehot.div(2)
  return continous_mol

def continous_mols_to_selfies(continous_mols, selfies_alphabet, int_mol):
   denormalized_data = continous_mols * 2
   quantized_data = torch.floor(denormalized_data)
   quantized_data = torch.clip(quantized_data, 0, 1)
   for mol in quantized_data:
    for letter in mol:
      if all(elem == 0 for elem in letter):
        letter[len(selfies_alphabet)] = 1
   selfies = [sf.encoding_to_selfies(mol.cpu().tolist(), int_mol, enc_type="one_hot") for mol in quantized_data]
   return selfies

def continous_mols_to_smiles(continous_mols, featurizer):
  denormalized_data = continous_mols * 2
  quantized_data = torch.floor(denormalized_data)
  quantized_data = torch.clip(quantized_data, 0, 1)
  for mol in quantized_data:
    for letter in mol:
      if all(elem == 0 for elem in letter):
        letter[-1] = 1
  smiles = [featurizer.untransform(mol.cpu().tolist()) for mol in quantized_data]
  return smiles


def get_valid_selfies(selfies):
  valid_selfies = []
  for _, selfie in enumerate(selfies):
    try:
      if Chem.MolFromSmiles(sf.decoder(selfie), sanitize=True) is not None:
         valid_selfies.append(selfie)
    except Exception:
      pass
  return valid_selfies

def get_valid_smiles(smiles):
  valid_smiles = []
  for _, smile in enumerate(smiles):
    try:
      if Chem.MolFromSmiles(smile, sanitize=True) is not None:
         valid_smiles.append(smile)
    except Exception:
      pass
  return valid_smiles

def get_valid_smiles_and_classes(smiles_to_split, classes):
  valid_smiles, valid_classes = [], []
  for idx, smiles in enumerate(smiles_to_split):
    try:
      if Chem.MolFromSmiles(smiles, sanitize=True) is not None:
          valid_smiles.append(smiles)
          valid_classes.append(classes[idx])
    except Exception:
      pass
  return valid_smiles, valid_classes

def split_selfies_and_classes(selfies_to_split, classes):
  valid_selfies, valid_classes = [], []
  for idx, selfies in enumerate(selfies_to_split):
    try:
      if Chem.MolFromSmiles(sf.decoder(selfies), sanitize=True) is not None:
          valid_selfies.append(selfies)
          valid_classes.append(classes[idx])
    except Exception:
      pass
  return valid_selfies, valid_classes

def selfies_to_mols(selfies_to_convert):
  valid_count = 0
  valid_selfies, invalid_selfies = [], []
  mols = []
  for idx, selfies in enumerate(selfies_to_convert):
    try:
      if Chem.MolFromSmiles(sf.decoder(selfies), sanitize=True) is not None:
          valid_count += 1
          valid_selfies.append(selfies)
          mols.append(Chem.MolFromSmiles(sf.decoder(selfies)))
      else:
        invalid_selfies.append(selfies)
    except Exception:
      pass
  return mols, valid_selfies, valid_count

def smiles_to_mols(smiles_to_convert):
  valid_count = 0
  valid_smiles, invalid_smiles = [], []
  mols = []
  for idx, smiles in enumerate(smiles_to_convert):
    try:
      if Chem.MolFromSmiles(smiles, sanitize=True) is not None:
          valid_count += 1
          valid_smiles.append(smiles)
          mols.append(Chem.MolFromSmiles(smiles))
      else:
        invalid_smiles.append(smiles)
    except Exception:
      pass
  return mols, valid_smiles, valid_count

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
  return torch.tensor(discretized_values), breakpoints

def canonicalize_smiles(smiles):
  smiles_canonicalized = [canonicalize(smile) for smile in smiles]
  return smiles_canonicalized

def canonicalize_one_smiles(smiles):
  return canonicalize(smiles)

def get_mols_properties(mols, prop):
  if prop == "Weight":
    molsWt = [Chem.Descriptors.MolWt(mol) for mol in mols]
    return molsWt
  elif prop == "LogP":
    molsLogP = [Chem.Descriptors.MolLogP(mol) for mol in mols]
    return molsLogP
  elif prop == "QED":
    molsQED = [Chem.QED.default(mol) for mol in mols]
    return molsQED
  elif prop == "SAS":
    molsSAS = [sascorer.calculateScore(mol) for mol in mols]
    return molsSAS

def get_selfies_properties(selfies, prop):
  props = []
  for _, selfie in enumerate(selfies):
    try:
      mol = Chem.MolFromSmiles(sf.decoder(selfie), sanitize=True)
      if mol is not None:
         if prop == "Weight":
            props.append(Chem.Descriptors.MolWt(mol))
         elif prop == "LogP":
            props.append(Chem.Descriptors.MolLogP(mol))
         elif prop == "QED":
            props.append(Chem.QED.default(mol))
         elif prop == "SAS":
            props.append(sascorer.calculateScore(mol))
    except Exception:
      pass
  return props

def get_smiles_properties(smiles, prop):
  props = []
  for _, smile in enumerate(smiles):
    try:
      mol = Chem.MolFromSmiles(smile, sanitize=True)
      if mol is not None:
         if prop == "Weight":
            props.append(Chem.Descriptors.MolWt(mol))
         elif prop == "LogP":
            props.append(Chem.Descriptors.MolLogP(mol))
         elif prop == "QED":
            props.append(Chem.QED.default(mol))
         elif prop == "SAS":
            props.append(sascorer.calculateScore(mol))
    except Exception:
      pass
  return props