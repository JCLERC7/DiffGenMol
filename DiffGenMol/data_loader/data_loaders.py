from datasets import load_dataset
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch.utils.data import Dataset, DataLoader
import torch
from torch import Tensor
import utils

# Class of selfies dataset
class DatasetSelfies(Dataset):
    def __init__(self, x: np.ndarray, y: Tensor, largest_selfie_len, selfies_alphabet, symbol_to_int):
        super().__init__()
        self.x = x
        self.y = y.clone()
        self.largest_selfie_len = largest_selfie_len
        self.selfies_alphabet = selfies_alphabet
        self.symbol_to_int = symbol_to_int

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {'continous_selfies': utils.selfies_to_continous_mols([self.x[idx]], self.largest_selfie_len, self.selfies_alphabet, self.symbol_to_int), 'classe': self.y[idx]}
    
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

class GuacamolDataLoader():
    def __init__(self, dataset_size, min_smiles_size, max_smiles_size, batch_size, num_classes, type_property, shuffle, config):
        self.config = config
        self.logger = config.get_logger('train')

        self.batch_size = batch_size
        self.num_classes = num_classes
        self.type_property = type_property

        # load dataset 
        self.logger.info('Loading dataset')
        dataset = load_dataset("jxie/guacamol")
        train_smiles_prep = [data['text'] for data in dataset['train']]

        if max_smiles_size > 0:
            train_smiles_prep = utils.by_size(train_smiles_prep, min_smiles_size ,max_smiles_size)
            
        df = pd.DataFrame(data={'smiles': train_smiles_prep})
        
        df = df[df["smiles"].str.contains('IH2') == False] # [IH2] don't work with selfies

        if dataset_size > 0:
            df = df[['smiles']].sample(dataset_size, random_state=42)
        else:
            df = df[['smiles']] 

        self.train_smiles = df['smiles']

        self.logger.info('Converting to selfies')
        self.train_selfies = utils.smiles_to_selfies(self.train_smiles)
        self.logger.info('Calculate selfies features')
        self.largest_selfie_len, self.selfies_alphabet, self.symbol_to_int, self.int_mol = utils.get_selfies_features(self.train_selfies)
        self.seq_length = self.largest_selfie_len * len(self.selfies_alphabet)
        self.nb_mols = len(self.train_selfies)
        self.logger.info(f'nb_mols: {self.nb_mols}')

        # original mols for similarity calculation
        self.train_mols, _, _ = utils.selfies_to_mols(self.train_selfies)

        # properties
        self.logger.info('Calculating molecules properties')
        self.continuous_properties = get_mols_properties(self.train_mols,self.type_property)
        self.train_classes, self.classes_breakpoints = utils.discretize_continuous_values(self.continuous_properties, num_classes)

        self.dataset = DatasetSelfies(self.train_selfies, torch.tensor(self.train_classes), self.largest_selfie_len, self.selfies_alphabet, self.symbol_to_int)
        # create dataloader
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
    
    def get_train_smiles(self):
        return self.train_smiles
    
    def get_train_mols(self):
        return self.train_mols
    
    def get_train_classes(self):
        return self.train_classes

    def get_num_classes(self):
        return self.num_classes
    
    def get_dataloader(self):
        return self.dataloader
    
    def get_seq_length(self):
        return self.seq_length
    
    def get_type_property(self):
        return self.type_property
    
    def get_classes_breakpoints(self):
        return self.classes_breakpoints

    def get_selfies_alphabet(self):
        return self.selfies_alphabet   

    def get_largest_selfie_len(self):
        return self.largest_selfie_len 

    def get_int_mol(self):
        return self.int_mol
