import torch
import torch.nn as nn

# ==========================================
# LE SENSEUR DU MILIEU (L'instinct de survie)
# ==========================================
class SenseurEnvironnement:
    @staticmethod
    def verifier_ressources():
        """Le samouraï ressent la VRAM du GPU"""
        if torch.cuda.is_available():
            vram_libre, _ = torch.cuda.mem_get_info()
            return vram_libre / 1e9
        return 8.0

# ==========================================
# L'ARCHITECTURE DU CERVEAU (De zéro)
# ==========================================
class SamouraiBlock(nn.Module):
    def __init__(self, dim_modele, nombre_tetes, taux_dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim_modele, nombre_tetes, batch_first=True, dropout=taux_dropout)
        self.reseau_pensee = nn.Sequential(
            nn.Linear(dim_modele, dim_modele * 4),
            nn.GELU(),
            nn.Dropout(taux_dropout),
            nn.Linear(dim_modele * 4, dim_modele)
        )
        self.norme1 = nn.LayerNorm(dim_modele)
        self.norme2 = nn.LayerNorm(dim_modele)

    def forward(self, x):
        taille = x.size(1)
        masque = torch.triu(torch.ones(taille, taille, device=x.device), diagonal=1).bool()
        attention_out, _ = self.attention(x, x, x, attn_mask=masque)
        x = self.norme1(x + attention_out)
        pensee_out = self.reseau_pensee(x)
        x = self.norme2(x + pensee_out)
        return x

class MonLLMSamourai(nn.Module):
    def __init__(self, taille_vocabulaire, dim_modele=384, nombre_blocs=6, nombre_tetes=6, taille_contexte=256):
        super().__init__()
        self.taille_contexte = taille_contexte
        self.embedding_mot = nn.Embedding(taille_vocabulaire, dim_modele)
        self.embedding_position = nn.Embedding(taille_contexte, dim_modele)
        
        self.blocs_samourai = nn.ModuleList([
            SamouraiBlock(dim_modele, nombre_tetes) for _ in range(nombre_blocs)
        ])
        
        self.norme_finale = nn.LayerNorm(dim_modele)
        self.sortie = nn.Linear(dim_modele, taille_vocabulaire)
        self._init_poids()

    def _init_poids(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        if x.size(1) > self.taille_contexte:
            x = x[:, -self.taille_contexte:]
            
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x = self.embedding_mot(x) + self.embedding_position(positions)
        
        # Le Bushido Adaptatif : L'IA ressent son environnement
        vram_libre = SenseurEnvironnement.verifier_ressources()
        
        if vram_libre < 2.0:
            # MODE SURVIE : On limite la réflexion
            x = self.blocs_samourai[0](x)
        else:
            # MODE COMBAT : Pleine puissance
            for bloc in self.blocs_samourai:
                x = bloc(x)
                
        x = self.norme_finale(x)
        return self.sortie(x)