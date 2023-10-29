#Tests pour le remplacement de
#onehots=sf.batch_selfies_to_flat_hot(selfies, symbol_to_int,largest_selfie_len)
import selfies as sf
import torch


def keys_int(symbol_to_int):
  d={}
  i=0
  for key in symbol_to_int.keys():
    d[i]=key
    i+=1
  return d

selfies = ["[C][F]", "[C][F]", "[C][F]", "c1ccccc1o"]
selfies_alphabet = sf.get_alphabet_from_selfies(selfies)
selfies_alphabet.add('[nop]')  # Add the "no operation" symbol as a padding character
selfies_alphabet.add('.') 
selfies_alphabet = list(sorted(selfies_alphabet))
largest_selfie_len = max(sf.len_selfies(s) for s in selfies)
symbol_to_int = dict((c, i) for i, c in enumerate(selfies_alphabet))
int_mol=keys_int(symbol_to_int)

onehots = list()
for selfie in selfies:
  one_hot = sf.selfies_to_encoding(selfie, symbol_to_int, largest_selfie_len, enc_type='one_hot')
  flattened = [elem for vec in one_hot for elem in vec]
  onehots.append(flattened)

print(f'onehots: {onehots}')

onehots = torch.zeros(len(selfies), largest_selfie_len, len(selfies_alphabet), dtype=torch.float)
for i, selfie in enumerate(selfies):
    one_hot = sf.selfies_to_encoding(selfie, symbol_to_int, largest_selfie_len, enc_type='one_hot')
    onehots[i, :, :] = torch.tensor(one_hot, dtype=torch.float32)
onehots_flat = onehots.view(len(selfies), -1)

print(f'onehots: {onehots_flat}')

input_tensor = onehots.view(len(selfies), -1)
torch.manual_seed(1)
noise_tensor = torch.rand(input_tensor.shape, dtype=torch.float32)
dequantized_onehots = input_tensor + noise_tensor
continous_mols = (dequantized_onehots - dequantized_onehots.min()) / (dequantized_onehots.max() - dequantized_onehots.min())

print(continous_mols)

input_tensor = onehots.view(len(selfies), -1)
torch.manual_seed(1)
dequantized_onehots = input_tensor.add(torch.rand(input_tensor.shape, dtype=torch.float32))
dequantized_onehots_min = dequantized_onehots.min()
dequantized_onehots_diff = dequantized_onehots.max() - dequantized_onehots_min
continous_mols = dequantized_onehots.sub(dequantized_onehots_min).div(dequantized_onehots_diff)

print(continous_mols)

