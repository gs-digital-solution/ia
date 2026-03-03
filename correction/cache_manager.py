"""
Module de gestion du cache pour les corrigés
Utilise Redis pour stocker les résultats IA et éviter les appels redondants
"""

import redis
import hashlib
import json
import logging
import time
from datetime import timedelta
from typing import Optional, Dict, Any
from django.conf import settings

logger = logging.getLogger(__name__)


class CorrigeCacheManager:
    """
    Gestionnaire de cache pour les corrigés IA

    Principe :
    1. On calcule une empreinte (hash) du contenu de l'exercice
    2. On vérifie si cette empreinte existe dans Redis
    3. Si oui, on retourne le corrigé stocké (économie d'appel API)
    4. Si non, on laisse l'API faire son travail et on stocke le résultat
    """

    def __init__(self):
        """Initialise la connexion Redis"""
        self.redis_client = None
        self._connect()

        # Durée de conservation des corrigés (30 jours par défaut)
        self.CACHE_TTL = getattr(settings, 'CORRIGE_CACHE_TTL', 30 * 24 * 60 * 60)

        # Version du cache (pour invalidation si on change le format)
        self.CACHE_VERSION = getattr(settings, 'CORRIGE_CACHE_VERSION', 1)

        logger.info(f"✅ Cache manager initialisé (TTL: {self.CACHE_TTL}s, v{self.CACHE_VERSION})")

    def _connect(self):
        """Établit la connexion Redis avec gestion d'erreur"""
        try:
            self.redis_client = redis.Redis(
                host=getattr(settings, 'REDIS_HOST', 'localhost'),
                port=getattr(settings, 'REDIS_PORT', 6379),
                db=getattr(settings, 'REDIS_CACHE_DB', 1),  # DB 1 pour le cache
                password=getattr(settings, 'REDIS_PASSWORD', None),
                socket_connect_timeout=2,
                socket_timeout=2,
                decode_responses=True,  # Important : retourne des strings directement
                retry_on_timeout=True
            )
            # Test de connexion
            self.redis_client.ping()
            logger.info("✅ Connexion Redis établie (cache)")
        except redis.ConnectionError as e:
            logger.error(f"❌ Connexion Redis échouée: {e}")
            self.redis_client = None
        except Exception as e:
            logger.error(f"❌ Erreur Redis inattendue: {e}")
            self.redis_client = None

    def _compute_fingerprint(self, texte_exercice: str, matiere_id: Optional[int] = None) -> str:
        """
        Calcule une empreinte unique de l'exercice

        Args:
            texte_exercice: Le texte de l'exercice
            matiere_id: ID de la matière (optionnel, pour plus de précision)

        Returns:
            str: Empreinte unique (hash MD5)

        Exemple:
            "Résoudre 2x+3=7" + matière=maths → "7d8f3e2a1b5c9f4d8e2a1b5c9f4d8e2a"
        """
        # Nettoyer le texte pour ignorer les différences mineures
        texte_normalise = self._normalize_text(texte_exercice)

        # LOG TEMPORAIRE pour voir le texte normalisé
        logger.info(f"🔍 TEXTE NORMALISÉ (début): {texte_normalise[:200]}")

        if matiere_id:
            texte_normalise = f"{texte_normalise}|matiere:{matiere_id}"
            logger.info(f"🔍 AVEC MATIÈRE: {matiere_id}")

        fingerprint = hashlib.md5(texte_normalise.encode('utf-8')).hexdigest()
        logger.info(f"🔍 FINGERPRINT: {fingerprint[:16]}...")
        return fingerprint

        # Ajouter la matière si fournie
        if matiere_id:
            texte_normalise = f"{texte_normalise}|matiere:{matiere_id}"

        # Calculer le hash
        fingerprint = hashlib.md5(texte_normalise.encode('utf-8')).hexdigest()

        logger.debug(f"🔑 Empreinte calculée: {fingerprint[:8]}... (texte: {len(texte_normalise)} chars)")
        return fingerprint

    def _normalize_text(self, texte: str) -> str:
        """
        Normalise le texte pour que des variations mineures donnent la même empreinte

        Supprime:
        - Espaces multiples
        - Retours à la ligne excessifs
        - Ponctuation en trop
        - Majuscules (optionnel)
        """
        import re

        # Remplacer les retours à la ligne multiples par un seul
        texte = re.sub(r'\n\s*\n', '\n', texte)

        # Supprimer les espaces en début/fin de ligne
        texte = '\n'.join(line.strip() for line in texte.split('\n'))

        # Remplacer les espaces multiples par un seul
        texte = re.sub(r' +', ' ', texte)

        # Option: ignorer la casse (tout en minuscules)
        # À activer si "Équation" et "équation" doivent être considérés identiques
        # texte = texte.lower()

        return texte.strip()

    def get(self, texte_exercice: str, matiere_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Récupère un corrigé du cache s'il existe

        Args:
            texte_exercice: Texte de l'exercice
            matiere_id: ID de la matière (optionnel)

        Returns:
            Le corrigé (dict) ou None si non trouvé
        """
        if not self.redis_client:
            logger.warning("⚠️ Redis indisponible, cache désactivé")
            return None

        try:
            # 1. Calculer l'empreinte
            fingerprint = self._compute_fingerprint(texte_exercice, matiere_id)

            # 2. Clé Redis complète avec version
            cache_key = f"corrige:v{self.CACHE_VERSION}:{fingerprint}"

            # 3. Récupérer depuis Redis
            start_time = time.time()
            cached_data = self.redis_client.get(cache_key)
            query_time = (time.time() - start_time) * 1000  # en ms

            if cached_data:
                # Désérialiser le JSON
                try:
                    result = json.loads(cached_data)
                    logger.info(f"✅ CACHE HIT! ({query_time:.1f}ms) - clé: {cache_key[:20]}...")

                    # Ajouter des métadonnées
                    result['_cache_hit'] = True
                    result['_cache_key'] = cache_key
                    result['_cache_age'] = self.redis_client.ttl(cache_key)

                    return result
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Erreur décodage JSON du cache: {e}")
                    # Supprimer l'entrée corrompue
                    self.redis_client.delete(cache_key)
                    return None
            else:
                logger.info(f"⏳ CACHE MISS ({query_time:.1f}ms) - clé: {cache_key[:20]}...")
                return None

        except redis.RedisError as e:
            logger.error(f"❌ Erreur Redis lors de la lecture: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Erreur inattendue lors de la lecture cache: {e}")
            return None

    def set(self, texte_exercice: str, resultat: Dict[str, Any],
            matiere_id: Optional[int] = None, ttl: Optional[int] = None) -> bool:
        """
        Stocke un corrigé dans le cache

        Args:
            texte_exercice: Texte de l'exercice
            resultat: Le corrigé à stocker (dict)
            matiere_id: ID de la matière (optionnel)
            ttl: Durée de conservation en secondes (défaut: self.CACHE_TTL)

        Returns:
            bool: True si stocké avec succès
        """
        if not self.redis_client:
            return False

        try:
            # 1. Calculer l'empreinte
            fingerprint = self._compute_fingerprint(texte_exercice, matiere_id)

            # 2. Clé Redis
            cache_key = f"corrige:v{self.CACHE_VERSION}:{fingerprint}"

            # 3. Préparer les données à stocker
            # On enlève les éventuelles métadonnées de cache
            data_to_store = {k: v for k, v in resultat.items()
                             if not k.startswith('_cache_')}

            # Ajouter un timestamp pour debug
            data_to_store['_cached_at'] = time.time()

            # 4. Sérialiser en JSON
            json_data = json.dumps(data_to_store, ensure_ascii=False)

            # 5. Stocker dans Redis avec expiration
            ttl = ttl or self.CACHE_TTL
            self.redis_client.setex(cache_key, ttl, json_data)

            logger.info(f"💾 CACHE STORE - clé: {cache_key[:20]}... (ttl: {ttl}s)")
            return True

        except redis.RedisError as e:
            logger.error(f"❌ Erreur Redis lors de l'écriture: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Erreur inattendue lors de l'écriture cache: {e}")
            return False

    def delete(self, texte_exercice: str, matiere_id: Optional[int] = None) -> bool:
        """Supprime un corrigé du cache"""
        if not self.redis_client:
            return False

        try:
            fingerprint = self._compute_fingerprint(texte_exercice, matiere_id)
            cache_key = f"corrige:v{self.CACHE_VERSION}:{fingerprint}"
            result = self.redis_client.delete(cache_key)
            logger.info(f"🗑️ CACHE DELETE: {cache_key[:20]}... (supprimé: {result})")
            return result > 0
        except Exception as e:
            logger.error(f"❌ Erreur suppression cache: {e}")
            return False

    def clear_all(self) -> bool:
        """Supprime TOUS les corrigés du cache (à utiliser avec précaution)"""
        if not self.redis_client:
            return False

        try:
            # Chercher toutes les clés de corrigés
            pattern = f"corrige:v{self.CACHE_VERSION}:*"
            keys = self.redis_client.keys(pattern)

            if keys:
                count = self.redis_client.delete(*keys)
                logger.warning(f"⚠️ CACHE CLEAR: {count} corrigés supprimés")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Erreur clear cache: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Retourne des statistiques sur le cache"""
        if not self.redis_client:
            return {"status": "redis_disconnected"}

        try:
            pattern = f"corrige:v{self.CACHE_VERSION}:*"
            keys = self.redis_client.keys(pattern)

            # Récupérer les infos Redis
            info = self.redis_client.info()

            return {
                "status": "connected",
                "total_cached": len(keys),
                "memory_used": info.get('used_memory_human', 'N/A'),
                "redis_version": info.get('redis_version', 'N/A'),
                "cache_version": self.CACHE_VERSION,
                "ttl_seconds": self.CACHE_TTL
            }
        except Exception as e:
            logger.error(f"❌ Erreur stats cache: {e}")
            return {"status": "error", "error": str(e)}


# Singleton pattern
_cache_manager = None


def get_cache_manager():
    """Retourne l'instance unique du cache manager"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CorrigeCacheManager()
    return _cache_manager


def with_cache(func):
    """
    Décorateur pour utiliser automatiquement le cache

    Utilisation:
        @with_cache
        def generer_corrige_par_exercice(texte_exercice, ..., demande=None):
            ...

    Le décorateur:
    1. Vérifie si le corrigé existe en cache
    2. Si oui, retourne immédiatement le corrigé
    3. Si non, appelle la fonction, puis stocke le résultat
    """
    from functools import wraps


    @wraps(func)
    def wrapper(texte_exercice, *args, **kwargs):
        # Récupérer la demande depuis les kwargs
        demande = kwargs.get('demande')

        # Essayer de récupérer le contenu nettoyé depuis la soumission
        contenu_nettoye = None
        if demande:
            # Chercher la soumission la plus récente pour cette demande
            soumission = demande.soumissionia_set.order_by('-date_creation').first()
            if soumission and soumission.resultat_json:
                contenu_nettoye = soumission.resultat_json.get('contenu_nettoye')

        # Si on a un contenu nettoyé, on l'utilise pour le cache
        texte_pour_cache = contenu_nettoye if contenu_nettoye else texte_exercice

        # Logger pour debug
        logger.info(f"🔍 Cache utilisant: {'contenu_nettoye' if contenu_nettoye else 'texte_brut'}")

        cache = get_cache_manager()
        matiere = kwargs.get('matiere')
        matiere_id = matiere.id if matiere else None

        # Vérifier le cache avec le bon texte
        cached_result = cache.get(texte_pour_cache, matiere_id)

        if cached_result:
            logger.info(f"🎯 CACHE HIT! (utilisé contenu nettoyé: {bool(contenu_nettoye)})")
            if isinstance(cached_result, dict):
                return cached_result.get('corrige_text', ''), cached_result.get('graphiques', [])
            return cached_result, []

        # Pas dans le cache, exécuter la fonction
        logger.info("🤖 CACHE MISS, exécution fonction originale")
        result = func(texte_exercice, *args, **kwargs)

        # Stocker dans le cache (avec le contenu nettoyé pour les prochains)
        if result and isinstance(result, tuple) and len(result) >= 1:
            to_cache = {
                'corrige_text': result[0],
                'graphiques': result[1] if len(result) > 1 else []
            }
            cache.set(texte_pour_cache, to_cache, matiere_id)

        return result

    return wrapper
