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

class GuacamolDataLoader():
    def __init__(self, batch_size, num_classes, type_property, shuffle):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.type_property = type_property

        # load dataset 
        self.original_selfies = utils.smiles_to_selfies(utils.get_smiles())
        self.dataset, self.selfies_alphabet, self.largest_selfie_len, self.int_mol, self.dequantized_onehots_min, self.dequantized_onehots_max = utils.selfies_to_continous_mols(self.original_selfies)
        self.seq_length = self.dataset.shape[1]
        
        # original mols for similarity calculation
        self.original_mols, _, _ = utils.selfies_to_mols(self.original_selfies)

        # properties
        self.continuous_properties = utils.get_mols_properties(self.original_mols,self.type_property)
        self.classes, self.classes_breakpoints = utils.discretize_continuous_values(self.continuous_properties, num_classes)

        self.dataset = Dataset1D(self.dataset[:, None, :].float(), torch.tensor(self.classes))
        # create dataloader
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
    
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