# Utiliser une image Python légère
FROM python:3.10-slim

# Définir le dossier de travail
WORKDIR /app

# 1. Copier les requirements d'abord (pour le cache Docker)
COPY requirements.txt .

# 2. Installer les dépendances
# --no-cache-dir garde l'image légère
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copier tout le reste du code (app.py, src/, etl/, etc.)
COPY . .

# 4. Configuration spécifique pour Hugging Face
# HF attend que l'app écoute sur le port 7860
ENV HOST=0.0.0.0
ENV LISTEN_PORT=7860
EXPOSE 7860

# 5. La commande de lancement
# On lance Chainlit sur le port 7860
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]