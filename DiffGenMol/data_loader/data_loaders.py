from datasets import load_dataset
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch.utils.data import Dataset, DataLoader
import torch
from torch import Tensor
import utils

# Class of custom dataset
class Dataset1D(Dataset):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__()
        self.x = x.clone()
        self.y = y.clone()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
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
    def __init__(self, dataset_size, max_smiles_size, batch_size, num_classes, type_property, shuffle, config):
        self.config = config
        self.logger = config.get_logger('train')

        self.batch_size = batch_size
        self.num_classes = num_classes
        self.type_property = type_property

        # load dataset 
        self.logger.info('Loading dataset')
        dataset = load_dataset("jxie/guacamol")
        train_smiles_complete = [data['text'] for data in dataset['train']]

        if max_smiles_size > 0:
            train_smiles_by_max_size = utils.by_max_size(train_smiles_complete, max_smiles_size)
        else:
            train_smiles_by_max_size = train_smiles_complete
            
        df = pd.DataFrame(data={'smiles': train_smiles_by_max_size})
        
        df = df[df["smiles"].str.contains("[IH2]") == False] # [IH2] don't convert in selfies

        if dataset_size > 0:
            data = df[['smiles']].sample(dataset_size, random_state=42)
        else:
            data = df[['smiles']] 

        self.train_smiles = data['smiles']
        self.logger.info('Converting to selfies')
        self.train_selfies = utils.smiles_to_selfies(self.train_smiles)
        self.logger.info('Converting to continuous mols')
        self.train_continous_mols, self.selfies_alphabet, self.largest_selfie_len, self.int_mol, self.dequantized_onehots_min, self.dequantized_onehots_max = utils.selfies_to_continous_mols(self.train_selfies)
        self.seq_length = self.train_continous_mols.shape[1]
        
        # original mols for similarity calculation
        self.train_mols, _, _ = utils.selfies_to_mols(self.train_selfies)

        # properties
        self.logger.info('Calculating properties')
        self.continuous_properties = get_mols_properties(self.train_mols,self.type_property)
        self.train_classes, self.classes_breakpoints = utils.discretize_continuous_values(self.continuous_properties, num_classes)

        self.dataset = Dataset1D(self.train_continous_mols[:, None, :].float(), torch.tensor(self.train_classes))
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
    
    def get_dequantized_onehots_min(self):
        return self.dequantized_onehots_min
            
    def get_dequantized_onehots_max(self):
        return self.dequantized_onehots_max