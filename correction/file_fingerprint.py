"""
Module pour calculer l'empreinte unique d'un fichier
"""

import hashlib
import os
import logging

logger = logging.getLogger(__name__)


def calculate_fingerprint(file_path, use_chunks=True, chunk_size=8192):
    """
    Calcule une empreinte unique d'un fichier basée sur son contenu

    Args:
        file_path: Chemin vers le fichier
        use_chunks: Utiliser la lecture par chunks (recommandé pour gros fichiers)
        chunk_size: Taille des chunks en octets (défaut: 8 Ko)

    Returns:
        str: Empreinte SHA256 du fichier
    """
    if not os.path.exists(file_path):
        logger.error(f"❌ Fichier non trouvé: {file_path}")
        return None

    try:
        taille = os.path.getsize(file_path)

        if use_chunks:
            # Méthode économique en mémoire
            hash_obj = hashlib.sha256()
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    hash_obj.update(chunk)
            hash_contenu = hash_obj.hexdigest()
        else:
            # Méthode simple (charge tout en mémoire)
            with open(file_path, 'rb') as f:
                contenu = f.read()
            hash_contenu = hashlib.sha256(contenu).hexdigest()

        # Combiner taille et hash du contenu pour plus de robustesse
        empreinte = hashlib.sha256(
            f"{taille}:{hash_contenu}".encode()
        ).hexdigest()

        logger.debug(f"🔑 Empreinte calculée: {empreinte[:16]}... (taille: {taille} octets)")
        return empreinte

    except Exception as e:
        logger.error(f"❌ Erreur calcul empreinte: {e}")
        return None


def calculate_fingerprint_simple(file_content):
    """
    Version simplifiée pour les données en mémoire
    """
    if isinstance(file_content, str):
        file_content = file_content.encode('utf-8')

    return hashlib.sha256(file_content).hexdigest()