import torch
import argparse
import os
from tokenizers import Tokenizer
from model import MonLLMSamourai

def parler_au_samourai(prompt, max_tokens=40):
    """Charge le Samouraï entraîné et lui demande de générer du code."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🗡️ Le Maître ordonne : '{prompt}'")

    # 1. Chargement des Yeux (Tokenizer)
    if not os.path.exists("tokenizer_samourai.json"):
        print("❌ Erreur : Le tokéniseur n'existe pas. Lancez 'train.py' d'abord.")
        return
        
    tokenizer = Tokenizer.from_file("tokenizer_samourai.json")
    TAILLE_VOCAB = tokenizer.get_vocab_size()

    # 2. Chargement du Cerveau Entraîné
    samourai = MonLLMSamourai(taille_vocabulaire=TAILLE_VOCAB).to(device)
    if not os.path.exists("samourai_cerveau.pth"):
        print("❌ Erreur : Le cerveau (samourai_cerveau.pth) est introuvable. Lancez 'train.py' d'abord.")
        return
        
    # On charge la mémoire de l'entraînement
    samourai.load_state_dict(torch.load("samourai_cerveau.pth", map_location=device))
    samourai.eval() # On le met en mode "Combat" (sans apprentissage, juste de la réflexion)

    # 3. La Génération (La Voix)
    ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    print("🥷 Le Samouraï réfléchit...")
    with torch.no_grad():
        for _ in range(max_tokens):
            # On coupe la mémoire si la phrase est trop longue pour la VRAM
            if input_ids.size(1) >= 256:
                input_ids = input_ids[:, -255:]
                
            # L'IA réfléchit au mot suivant
            logits = samourai(input_ids)
            prochain_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            
            # Si l'IA dit "Fin", on s'arrête
            if prochain_id.item() == tokenizer.token_to_id("[EOS]"):
                break
                
            # On ajoute le mot à la phrase
            input_ids = torch.cat([input_ids, prochain_id], dim=1)
            
    # On reconvertit les chiffres en vrai texte
    texte_genere = tokenizer.decode(input_ids[0].tolist())
    print(f"🏆 Le Samouraï a écrit :\n{texte_genere}")
    return texte_genere

# Permet de lancer le script depuis le terminal avec un argument
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parler au Samouraï LLM")
    parser.add_argument("--prompt", type=str, default="def", help="Le mot de départ à donner au Samouraï")
    args = parser.parse_args()
    parler_au_samourai(args.prompt)