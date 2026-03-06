"""
Module de rate limiting pour les appels DeepSeek
Utilise Redis pour limiter le nombre d'appels par utilisateur et globalement
"""

import redis
import time
import logging
from functools import wraps
from django.conf import settings

logger = logging.getLogger(__name__)


class DeepSeekRateLimiter:
    """
    Rate limiter avec deux niveaux :
    1. Par utilisateur : limite stricte (sliding window)
    2. Global : limite souple (fixed window)
    """

    def __init__(self):
        self.redis_client = None
        self._connect()
        self._load_scripts()

        # Configuration des limites (depuis settings ou valeurs par défaut)
        self.USER_LIMIT = getattr(settings, 'DEEPSEEK_USER_LIMIT', 2)  # 2 appels/min par user
        self.GLOBAL_LIMIT = getattr(settings, 'DEEPSEEK_GLOBAL_LIMIT', 20)  # 20 appels/min total

        logger.info(f"✅ Rate limiter initialisé: {self.USER_LIMIT}/user, {self.GLOBAL_LIMIT}/global")

    def _connect(self):
        """Établit la connexion Redis avec gestion d'erreur"""
        try:
            self.redis_client = redis.Redis(
                host=getattr(settings, 'REDIS_HOST', 'localhost'),
                port=getattr(settings, 'REDIS_PORT', 6379),
                db=getattr(settings, 'REDIS_RATE_LIMIT_DB', 2),  # DB 2 pour rate limiting
                password=getattr(settings, 'REDIS_PASSWORD', None),
                socket_connect_timeout=2,
                socket_timeout=2,
                decode_responses=True,
                retry_on_timeout=True
            )
            self.redis_client.ping()
            logger.info("✅ Connexion Redis établie (rate limiting)")
        except redis.ConnectionError as e:
            logger.error(f"❌ Connexion Redis échouée: {e}")
            self.redis_client = None
        except Exception as e:
            logger.error(f"❌ Erreur Redis inattendue: {e}")
            self.redis_client = None

    def _load_scripts(self):
        """Charge les scripts Lua pour les opérations atomiques"""
        self.sliding_window_script = """
        local key = KEYS[1]
        local limit = tonumber(ARGV[1])
        local window = tonumber(ARGV[2])
        local now = tonumber(redis.call('TIME')[1])

        -- Nettoyer les entrées plus vieilles que la fenêtre
        redis.call('ZREMRANGEBYSCORE', key, 0, now - window)

        -- Compter les entrées restantes
        local current = redis.call('ZCARD', key)

        if current < limit then
            -- Ajouter cette requête
            local unique_id = now .. ':' .. math.random(1000000)
            redis.call('ZADD', key, now, unique_id)
            redis.call('EXPIRE', key, window)
            return {1, limit - current - 1, 0}
        else
            -- Calculer le temps d'attente
            local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
            local retry_after = 0
            if #oldest > 0 then
                retry_after = (oldest[2] + window) - now
            end
            return {0, 0, math.ceil(retry_after)}
        end
        """

        if self.redis_client:
            try:
                self.sliding_window = self.redis_client.register_script(
                    self.sliding_window_script
                )
                logger.info("✅ Scripts Lua chargés avec succès")
            except Exception as e:
                logger.error(f"❌ Erreur chargement scripts Lua: {e}")
                self.sliding_window = None

    def _get_user_id(self, demande):
        """Extrait l'ID utilisateur de manière sécurisée"""
        try:
            if demande and hasattr(demande, 'user') and demande.user:
                return f"user:{demande.user.id}"
        except:
            pass
        return "anonymous"

    def can_call(self, demande=None, user_id=None):
        """
        Vérifie si un appel DeepSeek est autorisé

        Returns:
            tuple: (autorisé, message, retry_after)
        """
        if not self.redis_client or not self.sliding_window:
            logger.warning("⚠️ Redis indisponible - mode dégradé (appels autorisés)")
            return True, "Redis indisponible (mode dégradé)", 0

        try:
            # Déterminer l'ID utilisateur
            if user_id:
                uid = f"user:{user_id}"
            else:
                uid = self._get_user_id(demande)

            # 1) Rate limiting par utilisateur
            user_key = f"deepseek:user:{uid}"
            result = self.sliding_window(
                keys=[user_key],
                args=[self.USER_LIMIT, 60]
            )

            autorise = result[0] == 1
            retry_after = result[2]

            if not autorise:
                logger.warning(f"⛔ Rate limit user {uid}: attend {retry_after}s")
                return False, f"Limite utilisateur atteinte. Réessayez dans {retry_after}s", retry_after

            # 2) Rate limiting global (fixed window simple)
            current_minute = time.strftime("%Y-%m-%d-%H:%M")
            global_key = f"deepseek:global:{current_minute}"

            global_calls = self.redis_client.incr(global_key)
            if global_calls == 1:
                self.redis_client.expire(global_key, 60)

            if global_calls > self.GLOBAL_LIMIT:
                logger.warning(f"⛔ Rate limit global: {global_calls}/{self.GLOBAL_LIMIT}")
                return False, "Limite globale atteinte, réessayez dans 60s", 60

            logger.debug(f"✅ Appel autorisé: user {uid}, global {global_calls}/{self.GLOBAL_LIMIT}")
            return True, "OK", 0

        except redis.RedisError as e:
            logger.error(f"❌ Erreur Redis: {e}")
            return True, f"Erreur Redis: {e}", 0
        except Exception as e:
            logger.error(f"❌ Erreur rate limiter: {e}")
            return True, f"Erreur: {e}", 0

    def get_remaining_quota(self, demande=None, user_id=None):
        """Retourne le quota restant pour l'utilisateur"""
        if not self.redis_client:
            return self.USER_LIMIT

        try:
            if user_id:
                uid = f"user:{user_id}"
            else:
                uid = self._get_user_id(demande)

            user_key = f"deepseek:user:{uid}"
            now = time.time()
            current = self.redis_client.zcount(user_key, now - 60, now)
            return max(0, self.USER_LIMIT - current)
        except Exception as e:
            logger.error(f"❌ Erreur get_quota: {e}")
            return self.USER_LIMIT


# Singleton pattern
_rate_limiter = None


def get_rate_limiter():
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = DeepSeekRateLimiter()
    return _rate_limiter


def rate_limit_decorator(func):
    """
    Décorateur pour appliquer le rate limiting automatiquement
    """

    @wraps(func)
    def wrapper(texte_exercice, *args, **kwargs):
        demande = kwargs.get('demande')

        limiter = get_rate_limiter()
        allowed, message, retry_after = limiter.can_call(demande)

        if not allowed:
            error_msg = f"Rate limit: {message}"
            logger.error(error_msg)
            raise Exception(error_msg)

        return func(texte_exercice, *args, **kwargs)

    return wrapper