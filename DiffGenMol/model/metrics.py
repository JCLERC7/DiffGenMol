import torch
import utils
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Fingerprints.FingerprintMols import FingerprintMol
from rdkit.DataStructs import FingerprintSimilarity
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # suppress error messages
from guacamol.distribution_learning_benchmark import ValidityBenchmark, UniquenessBenchmark, NoveltyBenchmark, \
    KLDivBenchmark
from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from typing import List


class MockGenerator(DistributionMatchingGenerator):
    """
    Mock generator that returns pre-defined molecules,
    possibly split in several calls
    """

    def __init__(self, molecules: List[str]) -> None:
        self.molecules = molecules
        self.cursor = 0

    def generate(self, number_samples: int) -> List[str]:
        end = self.cursor + number_samples

        sampled_molecules = self.molecules[self.cursor:end]
        self.cursor = end
        return sampled_molecules

def validity_score(smiles):
    with torch.no_grad():
      generator = MockGenerator(smiles)
      benchmark = ValidityBenchmark(number_samples=len(smiles))
    return benchmark.assess_model(generator).score

def uniqueness_score(smiles):
    with torch.no_grad():
      generator = MockGenerator(smiles)
      benchmark = UniquenessBenchmark(number_samples=len(smiles))
    return benchmark.assess_model(generator).score

def novelty_score(gen_smiles, train_smiles):
    with torch.no_grad():
      generator = MockGenerator(gen_smiles)
      benchmark = NoveltyBenchmark(number_samples=len(gen_smiles), training_set=train_smiles)
    return benchmark.assess_model(generator).score

def KLdiv_score(gen_smiles, train_smiles):
    with torch.no_grad():
      generator = MockGenerator(gen_smiles)
      benchmark = KLDivBenchmark(number_samples=len(gen_smiles), training_set=train_smiles)
      result = benchmark.assess_model(generator)
    return result.score



def accuracy_valid_conversion_selfies_to_smiles(selfies):
    with torch.no_grad():
        _, _, valid_count = utils.selfies_to_mols(selfies)
    return valid_count / len(selfies)

def accuracy_match_classes(mols, classes, prop, num_classes, breakpoints):
  with torch.no_grad():
    if prop == "Weight":
        prop_values = [Chem.Descriptors.MolWt(mol) for mol in mols]
    elif prop == "LogP":
        prop_values = [Chem.Descriptors.MolLogP(mol) for mol in mols]
    elif prop == "QED":
        prop_values = [Chem.QED.default(mol) for mol in mols]
    true_classes, _ = utils.discretize_continuous_values(prop_values, num_classes, breakpoints)
    nb_diff = 0
    for c1, c2 in zip(classes, true_classes):
        if c1 != c2:
            nb_diff += 1
  return (len(classes)-nb_diff)/len(classes)

# Calculate similarity
def tanimoto_similarity(database_mols, query_mol):
  """Compare generated molecules to database by Tanimoto similarity."""
  with torch.no_grad():
    # convert Mol to datastructure type
    fps = [FingerprintMol(m) for m in database_mols]
    
    # set a query molecule to compare against database
    query = FingerprintMol(query_mol)
    
    similarities = []
    
    # loop through to find Tanimoto similarity
    for idx, f in enumerate(fps):
        # tuple: (idx, similarity)
        similarities.append((idx, FingerprintSimilarity(query, f)))
    
    # sort sim using the similarities
    similarities.sort(key=lambda x:x[1], reverse=True)
  
  return similarities

def top3_tanimoto_similarity_score(gen_mols, train_mols):
    with torch.no_grad():
      scores = 0
      for gen_mol in gen_mols:
         tanimoto_scores = tanimoto_similarity(train_mols, gen_mol)
         scores_list = [tanimoto_score[1] for tanimoto_score in tanimoto_scores[:3]]
         scores += sum(scores_list)
    return scores / (len(gen_mols)*3)

def top3_tanimoto_similarity_score_cond(gen_mols, gen_classes, train_mols, train_classes, nb_classes):
   with torch.no_grad():
      scores = 0
      for classe in range(nb_classes):
        gen_mols_cond = [gen_mols[i] for i in range(len(gen_classes)) if gen_classes[i] == classe]
        train_mols_cond = [train_mols[i] for i in range(len(train_classes)) if train_classes[i] == classe]
        scores += top3_tanimoto_similarity_score(gen_mols_cond, train_mols_cond)
   return scores / nb_classes

def KLdiv_score_cond(gen_smiles, gen_classes, train_smiles, train_classes, nb_classes):
   with torch.no_grad():
      scores = 0
      for classe in range(nb_classes):
        gen_smiles_cond = [gen_smiles[i] for i in range(len(gen_classes)) if gen_classes[i] == classe]
        train_smiles_cond = [train_smiles[i] for i in range(len(train_classes)) if train_classes[i] == classe]
        scores += KLdiv_score(gen_smiles_cond, train_smiles_cond)
   return scores / nb_classes