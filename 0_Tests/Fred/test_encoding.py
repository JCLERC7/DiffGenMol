import numpy as np

def encode_smiles(smiles, alphabet):
    """
    Encode une chaîne de SMILES en one-hot à l'aide d'un alphabet spécifié.

    Paramètres :
    - smiles : La chaîne de SMILES à encoder.
    - alphabet : La liste des caractères de l'alphabet.

    Retourne :
    - one_hot_encoded : Un tableau numpy représentant l'encodage one-hot.
    """

    # Créer un dictionnaire de correspondance entre les caractères de l'alphabet et leurs indices.
    char_to_index = {char: i for i, char in enumerate(alphabet)}

    # Initialiser un tableau numpy de la taille de l'alphabet avec des zéros.
    one_hot_encoded = np.zeros((len(smiles), len(alphabet)))

    # Encoder chaque caractère de la chaîne de SMILES en one-hot.
    i = 0
    while i < len(smiles):
        char = smiles[i]
        if i + 1 < len(smiles):
            # Vérifier si le caractère suivi forme un multi-caractère présent dans l'alphabet.
            next_char = smiles[i:i+2]
            if next_char in char_to_index:
                one_hot_encoded[i, char_to_index[next_char]] = 1
                i += 2
                continue

        # Si ce n'est pas un multi-caractère, encoder normalement.
        if char in char_to_index:
            one_hot_encoded[i, char_to_index[char]] = 1

        i += 1

    return one_hot_encoded

# Exemple d'utilisation :
smiles_example = "[OH+]"
alphabet_example = ['#', '(', ')', '-', '.', '/', '1', '2', '3', '4', '5', '6', '=', 'C', 'Cn', 'F', 'N', 'O',
                    '[C@@H]', '[C@@]', '[C@H+]', '[C@H]', '[C@]', '[CH+]', '[CH-]', '[CH2+]', '[CH2-]', '[CH]',
                    '[H]', '[N+]', '[N@@H+]', '[N@H+]', '[NH+]', '[NH-]', '[NH2+]', '[NH3+]', '[O-]', '[OH+]',
                    '[cH+]', '[cH-]', '[n+]', '[nH+]', '[nH]', '\\', 'c', 'n', 'o']

# Exemple d'utilisation avec les nouveaux SMILES
smiles_examples = ['CCC', 'C[C@H+]1[n+]']

for smiles in smiles_examples:
    result = encode_smiles(smiles, alphabet_example)
    print(f"SMILES: {smiles}")
    print("One-Hot Encoding:")
    print(result)
    print("-" * 50)