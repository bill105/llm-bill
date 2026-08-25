from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def entrainer_et_sauvegarder_tokenizer():
    """Crée le dictionnaire (Les Yeux) du Samouraï à partir de zéro."""
    print("🏋️ Entraînement du Tokéniseur (Les Yeux) sur le français et le code...")
    
    # 1. Création du modèle BPE (la même technologie que GPT-4)
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    # 2. Configuration du dictionnaire (5000 mots/symboles)
    trainer = BpeTrainer(
        vocab_size=5000,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    )
    
    # 3. Création du dataset de base (français + code)
    # (Pour la révolution, on remplacera ça plus tard par un gros fichier de données)
    donnees = [
        "def additionner(a, b):\n    return a + b\n",
        "Créez une fonction pour calculer la vitesse.",
        "variable = int(input('Entrez un nombre'))",
        "Le samouraï frappe le bois avec précision.",
        "for i in range(10):\n    print('Bonjour le monde')",
        "class Guerrier:\n    def __init__(self):\n        self.vie = 100",
    ]
    
    # On sauvegarde les données dans un fichier texte
    with open("dataset.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(donnees))
        
    # 4. Entraînement du tokéniseur sur ce fichier
    tokenizer.train(files=["dataset.txt"], trainer=trainer)
    
    # 5. Sauvegarde du dictionnaire
    tokenizer.save("tokenizer_samourai.json")
    print("✅ Tokéniseur sauvegardé sous 'tokenizer_samourai.json' !")

# Permet de lancer la fonction si on exécute le fichier directement
if __name__ == "__main__":
    entrainer_et_sauvegarder_tokenizer()