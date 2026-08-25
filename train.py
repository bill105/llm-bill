import torch
import torch.nn as nn
from torch.optim import AdamW
import ast
import os

# On importe nos propres fichiers créés juste avant
from model import MonLLMSamourai
from tokenizer import entrainer_et_sauvegarder_tokenizer
from tokenizers import Tokenizer

# ==========================================
# LE BUSHIDO (Votre méthode AÉCS)
# ==========================================
def refuteur_neuro_symbolique(code_genere):
    """Le système d'immunité de l'IA : détruit les hallucinations illogiques"""
    red_flags = ["sans raison", "magie", "tomber vers le haut"]
    for flaw in red_flags:
        if flaw in code_genere.lower():
            return -10.0  # Pénalité maximale
    try:
        ast.parse(code_genere)
        return 2.0  # Récompense : le code est syntaxiquement parfait
    except:
        return -1.0  # Pénalité légère : code cassé ou incomplet

def lancer_entrainement():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🗡️ Arme utilisée : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 1. Préparation des Yeux (Si le tokéniseur n'existe pas, on le crée)
    if not os.path.exists("tokenizer_samourai.json"):
        entrainer_et_sauvegarder_tokenizer()
    tokenizer = Tokenizer.from_file("tokenizer_samourai.json")

    # 2. Préparation des Données (Le Dojo)
    with open("dataset.txt", "r", encoding="utf-8") as f:
        texte = f.read()
    ids = tokenizer.encode(texte).ids
    data_tensor = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    # 3. Naissance du Cerveau
    TAILLE_VOCAB = tokenizer.get_vocab_size()
    samourai = MonLLMSamourai(taille_vocabulaire=TAILLE_VOCAB).to(device)
    print(f"🥷 Samouraï créé avec {sum(p.numel() for p in samourai.parameters())/1e6:.2f}M paramètres.")

    # 4. La boucle d'entraînement AÉCS
    optimiseur = AdamW(samourai.parameters(), lr=3e-4)
    samourai.train()
    loss_function = nn.CrossEntropyLoss()

    print("⚔️ Intégration de la méthode AÉCS (Apprentissage Contre-Factuel)...")
    for epoch in range(1, 51):
        x, y = data_tensor[:, :-1], data_tensor[:, 1:]
        preds = samourai(x)
        perte_prob = loss_function(preds.view(-1, TAILLE_VOCAB), y.view(-1))
        
        # LA RÉVOLUTION : Le réfutateur juge la pensée de l'IA en temps réel
        ids_pred = torch.argmax(preds, dim=-1)[0]
        texte_gen = tokenizer.decode(ids_pred.tolist())
        score = refuteur_neuro_symbolique(texte_gen)
        
        # La formule de perte AÉCS
        perte_aecs = perte_prob - (0.5 * score)
        
        optimiseur.zero_grad()
        perte_aecs.backward()
        optimiseur.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Tour {epoch}/50 | Erreur Standard : {perte_prob.item():.4f} | Score Logique AÉCS : {score}")

    # 5. Sauvegarde du cerveau
    torch.save(samourai.state_dict(), "samourai_cerveau.pth")
    print("✅ Entraînement terminé ! Le cerveau 'samourai_cerveau.pth' est sauvegardé.")

if __name__ == "__main__":
    lancer_entrainement()