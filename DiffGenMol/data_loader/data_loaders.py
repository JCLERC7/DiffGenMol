import csv
from datasets import load_dataset
import deepchem as dc
from multiprocessing import cpu_count
import numpy as np
import os
import pandas as pd
import pickle
from rdkit import Chem
from rdkit.Chem import Descriptors
import selfies as sf
from torch.utils.data import Dataset, DataLoader
import torch
from torch import Tensor
import utils

# Class of selfies dataset
class DatasetSelfies(Dataset):
    def __init__(self, x: np.ndarray, y: Tensor, largest_selfie_len, selfies_alphabet, symbol_to_int):
        super().__init__()
        self.x = x
        self.y = y
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
    except Exception:
      pass
  return props

class QM9DataLoader():
    def __init__(self, dataset_size, min_smiles_size, max_smiles_size, batch_size, num_classes, type_property, shuffle, config):
        self.config = config
        self.logger = config.get_logger('train')

        self.batch_size = batch_size
        self.num_classes = num_classes
        self.type_property = type_property

        # Load smiles dataset
        dataset_smiles_pickle = os.path.join(self.config.dataset_dir, 'dataset_QM9_smiles.pickle')
        if os.path.isfile(dataset_smiles_pickle):
            print("Loading existing smiles dataset")
            self.train_smiles =  pd.read_pickle(dataset_smiles_pickle)
        else:
            self.logger.info('Loading dataset')
            _, datasets, _ = dc.molnet.load_qm9(featurizer='ECFP')
            train_smiles_prep = pd.DataFrame(data={'smiles': datasets[0].ids})

            if max_smiles_size > 0:
                train_smiles_prep = utils.by_size(train_smiles_prep['smiles'], min_smiles_size ,max_smiles_size)
                
            df = pd.DataFrame(data={'smiles': train_smiles_prep})

            if dataset_size > 0:
                df = df[['smiles']].sample(dataset_size, random_state=42)
            else:
                df = df[['smiles']] 

            self.train_smiles = df['smiles']
            self.train_smiles.to_pickle(dataset_smiles_pickle)
            print("Smiles dataset saved")

        # Convert to Selfies
        dataset_selfies_pickle = os.path.join(self.config.dataset_dir, 'dataset_QM9_selfies.pickle')
        if os.path.isfile(dataset_selfies_pickle):
            print("Loading existing selfies conversion")
            with open(dataset_selfies_pickle, 'rb') as f:
                self.train_selfies = pickle.load(f)
        else:
            self.logger.info('Converting to selfies')
            self.train_selfies = utils.get_valid_selfies(utils.smiles_to_selfies(self.train_smiles, self.__class__.__name__))
            with open(dataset_selfies_pickle, 'wb') as f:
                pickle.dump(self.train_selfies, f)
            print("Selfies conversion saved")
        
        # Calculation Selfies features (alphabet and dico)
        self.logger.info('Calculate selfies features')
        self.largest_selfie_len, self.selfies_alphabet, self.symbol_to_int, self.int_mol = utils.get_selfies_features(self.train_selfies)
        self.seq_length = self.largest_selfie_len * len(self.selfies_alphabet)
        self.nb_mols = len(self.train_selfies)
        self.logger.info(f'nb_mols: {self.nb_mols}')

        # Calculation classes
        classes_pickle = os.path.join(self.config.dataset_dir, 'dataset_QM9_classes_' + self.type_property + '.pickle')
        classes_breakpoints_pickle = os.path.join(self.config.dataset_dir, 'dataset_QM9_classes_' + self.type_property + '_breakpoints.pickle')

        if os.path.isfile(classes_pickle) and os.path.isfile(classes_breakpoints_pickle):
            print("Loading existing classes")
            with open(classes_pickle, 'rb') as f:
                self.train_classes = pickle.load(f)
            with open(classes_breakpoints_pickle, 'rb') as f:
                self.classes_breakpoints = pickle.load(f)
        else:
            self.logger.info('Calculating and discretize molecules properties (classes)')
            self.train_classes, self.classes_breakpoints = utils.discretize_continuous_values(get_selfies_properties(self.train_selfies,self.type_property), self.num_classes)
            with open(classes_pickle, 'wb') as f:
                pickle.dump(self.train_classes, f)
            with open(classes_breakpoints_pickle, 'wb') as f:
                pickle.dump(self.classes_breakpoints, f)
            print("Classes and classes breakpoints saved")

        self.logger.info('Creating DatasetSelfies')
        self.dataset = DatasetSelfies(self.train_selfies, self.train_classes, self.largest_selfie_len, self.selfies_alphabet, self.symbol_to_int)
        # create dataloader
        self.logger.info('Creating DatasetLoader')
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, pin_memory = True, num_workers = cpu_count())
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_train_smiles(self):
        return self.train_smiles
    
    def get_nb_mols(self):
        return self.nb_mols
    
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

class GuacamolDataLoader():
    def __init__(self, dataset_size, min_smiles_size, max_smiles_size, batch_size, num_classes, type_property, shuffle, config):
        self.config = config
        self.logger = config.get_logger('train')

        self.batch_size = batch_size
        self.num_classes = num_classes
        self.type_property = type_property

        # Load smiles dataset
        dataset_smiles_pickle = os.path.join(self.config.dataset_dir, 'dataset_guacamol_smiles.pickle')
        if os.path.isfile(dataset_smiles_pickle):
            print("Loading existing smiles dataset")
            self.train_smiles =  pd.read_pickle(dataset_smiles_pickle)
        else:
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
            self.train_smiles.to_pickle(dataset_smiles_pickle)
            print("Smiles dataset saved")

        # Convert to Selfies
        dataset_selfies_pickle = os.path.join(self.config.dataset_dir, 'dataset_guacamol_selfies.pickle')
        if os.path.isfile(dataset_selfies_pickle):
            print("Loading existing selfies conversion")
            with open(dataset_selfies_pickle, 'rb') as f:
                self.train_selfies = pickle.load(f)
        else:
            self.logger.info('Converting to selfies')
            self.train_selfies = utils.get_valid_selfies(utils.smiles_to_selfies(self.train_smiles, self.__class__.__name__))
            with open(dataset_selfies_pickle, 'wb') as f:
                pickle.dump(self.train_selfies, f)
            print("Selfies conversion saved")
        
        # Calculation Selfies features (alphabet and dico)
        self.logger.info('Calculate selfies features')
        self.largest_selfie_len, self.selfies_alphabet, self.symbol_to_int, self.int_mol = utils.get_selfies_features(self.train_selfies)
        self.seq_length = self.largest_selfie_len * len(self.selfies_alphabet)
        self.nb_mols = len(self.train_selfies)
        self.logger.info(f'nb_mols: {self.nb_mols}')

        # Calculation classes
        classes_pickle = os.path.join(self.config.dataset_dir, 'dataset_guacamol_classes_' + self.type_property + '.pickle')
        classes_breakpoints_pickle = os.path.join(self.config.dataset_dir, 'dataset_guacamol_classes_' + self.type_property + '_breakpoints.pickle')

        if os.path.isfile(classes_pickle) and os.path.isfile(classes_breakpoints_pickle):
            print("Loading existing classes")
            with open(classes_pickle, 'rb') as f:
                self.train_classes = pickle.load(f)
            with open(classes_breakpoints_pickle, 'rb') as f:
                self.classes_breakpoints = pickle.load(f)
        else:
            self.logger.info('Calculating and discretize molecules properties (classes)')
            self.train_classes, self.classes_breakpoints = utils.discretize_continuous_values(get_selfies_properties(self.train_selfies,self.type_property), self.num_classes)
            with open(classes_pickle, 'wb') as f:
                pickle.dump(self.train_classes, f)
            with open(classes_breakpoints_pickle, 'wb') as f:
                pickle.dump(self.classes_breakpoints, f)
            print("Classes and classes breakpoints saved")

        self.logger.info('Creating DatasetSelfies')
        self.dataset = DatasetSelfies(self.train_selfies, self.train_classes, self.largest_selfie_len, self.selfies_alphabet, self.symbol_to_int)
        # create dataloader
        self.logger.info('Creating DatasetLoader')
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, pin_memory = True, num_workers = cpu_count())
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_train_smiles(self):
        return self.train_smiles
    
    def get_nb_mols(self):
        return self.nb_mols
    
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

class ZINCDataLoader():
    def __init__(self, dataset_size, min_smiles_size, max_smiles_size, batch_size, num_classes, type_property, shuffle, config):
        self.config = config
        self.logger = config.get_logger('train')

        self.batch_size = batch_size
        self.num_classes = num_classes
        self.type_property = type_property

        # Load smiles dataset
        self.logger.info('Loading dataset')
        dataset = pd.read_csv('data_loader/datasets/250k_rndm_zinc_drugs_clean_3.csv')
        train_smiles_prep = [data for data in dataset['smiles']]

        if max_smiles_size > 0:
            train_smiles_prep = utils.by_size(train_smiles_prep, min_smiles_size ,max_smiles_size)
            
        df = pd.DataFrame(data={'smiles': train_smiles_prep})
        
        if dataset_size > 0:
            df = df[['smiles']].sample(dataset_size, random_state=42)
        else:
            df = df[['smiles']] 

        self.train_smiles = df['smiles']

        print(len(self.train_smiles))

        # Convert to Selfies
        dataset_selfies_pickle = os.path.join(self.config.dataset_dir, 'dataset_ZINC_selfies.pickle')
        if os.path.isfile(dataset_selfies_pickle):
            print("Loading existing selfies conversion")
            with open(dataset_selfies_pickle, 'rb') as f:
                self.train_selfies = pickle.load(f)
        else:
            self.logger.info('Converting to selfies')
            self.train_selfies = utils.get_valid_selfies(utils.smiles_to_selfies(self.train_smiles, self.__class__.__name__))
            with open(dataset_selfies_pickle, 'wb') as f:
                pickle.dump(self.train_selfies, f)
            print("Selfies conversion saved")
        
        # Calculation Selfies features (alphabet and dico)
        self.logger.info('Calculate selfies features')
        self.largest_selfie_len, self.selfies_alphabet, self.symbol_to_int, self.int_mol = utils.get_selfies_features(self.train_selfies)
        self.seq_length = self.largest_selfie_len * len(self.selfies_alphabet)
        self.nb_mols = len(self.train_selfies)
        self.logger.info(f'nb_mols: {self.nb_mols}')

        # Debug
        print(self.selfies_alphabet)

        # Calculation classes
        classes_pickle = os.path.join(self.config.dataset_dir, 'dataset_ZINC_classes_' + self.type_property + '.pickle')
        classes_breakpoints_pickle = os.path.join(self.config.dataset_dir, 'dataset_ZINC_classes_' + self.type_property + '_breakpoints.pickle')

        if os.path.isfile(classes_pickle) and os.path.isfile(classes_breakpoints_pickle):
            print("Loading existing classes")
            with open(classes_pickle, 'rb') as f:
                self.train_classes = pickle.load(f)
            with open(classes_breakpoints_pickle, 'rb') as f:
                self.classes_breakpoints = pickle.load(f)
        else:
            self.logger.info('Calculating and discretize molecules properties (classes)')
            self.train_classes, self.classes_breakpoints = utils.discretize_continuous_values(get_selfies_properties(self.train_selfies,self.type_property), self.num_classes)
            with open(classes_pickle, 'wb') as f:
                pickle.dump(self.train_classes, f)
            with open(classes_breakpoints_pickle, 'wb') as f:
                pickle.dump(self.classes_breakpoints, f)
            print("Classes and classes breakpoints saved")

        self.logger.info('Creating DatasetSelfies')
        self.dataset = DatasetSelfies(self.train_selfies, self.train_classes, self.largest_selfie_len, self.selfies_alphabet, self.symbol_to_int)
        # create dataloader
        self.logger.info('Creating DatasetLoader')
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, pin_memory = True, num_workers = cpu_count())
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_train_smiles(self):
        return self.train_smiles
    
    def get_nb_mols(self):
        return self.nb_mols
    
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