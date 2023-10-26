from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Fingerprints.FingerprintMols import FingerprintMol
from rdkit.DataStructs import FingerprintSimilarity
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # suppress error messages

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
  
# Calculate similarity
def tanimoto_similarity(database_mols, query_mol):
  """Compare generated molecules to database by Tanimoto similarity."""
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

def discretize_continuous_values(values, num_classes, breakpoints = None):
  if breakpoints is None:
    sorted_values = sorted(values)
    breakpoints = [sorted_values[i * len(values) // num_classes] for i in range(1, num_classes)]
  discretized_values = []
  for value in values:
      class_value = sum(value > breakpoint for breakpoint in breakpoints)
      discretized_values.append(class_value)
  return discretized_values, breakpoints

def generate_mols_match_classes(mols, classes, prop, num_classes, breakpoints):
  if prop == "Weight":
    prop_values = [Chem.Descriptors.MolWt(mol) for mol in mols]
  elif prop == "LogP":
    prop_values = [Chem.Descriptors.MolLogP(mol) for mol in mols]
  elif prop == "QED":
    prop_values = [Chem.QED.default(mol) for mol in mols]
  true_classes, _ = discretize_continuous_values(prop_values, num_classes, breakpoints)
  print('Classes:')
  print(classes)
  print('True Classes:')
  print(true_classes)
  nb_diff = 0
  for c1, c2 in zip(classes, true_classes):
    if c1 != c2:
      nb_diff += 1
  print(nb_diff)
  return (len(classes)-nb_diff)/len(classes)*100