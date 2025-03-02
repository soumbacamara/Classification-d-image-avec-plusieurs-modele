import torch
from mlp import *
# Paramètres du modèle
input_size = 28 * 28        # Par exemple pour des images de taille 28x28 (comme MNIST)
hidden_sizes = [64, 32]       # Deux couches cachées de tailles 64 et 32
num_classes = 10              # Par exemple pour la classification de 10 classes
activation = 'relu'           # Choix de l'activation

# Création du modèle
model = MLP(input_size, hidden_sizes, num_classes, activation)

# Création d'un batch d'images factices
# Ici, des images en niveaux de gris de taille 28x28, batch de 8
dummy_images = torch.randn(8, 1, 28, 28)

# Passage en avant (forward pass)
logits = model(dummy_images)

# Affichage de la forme de la sortie
print("Shape of output logits:", logits.shape)  # Devrait afficher torch.Size([8, 10])
