import os

def get_provider_config_for_method(payment_method):
    """
    Récupère la configuration sensitive depuis les variables d'env
    en fonction du payment_method (Touchpay ou Campay).
    """
    # Iso code pays (fallback 'CMR')
    country = payment_method.pays.code.upper() if payment_method.pays else "CMR"
    code   = payment_method.code.upper()

    # --- 1) Touchpay ---
    if code.startswith("TOUCHPAY"):
        # URL globale pour tous les pays
        base_url = os.getenv("TOUCHPAY_BASE_URL", "https://apidist.gutouch.net/apidist")
        prefix = f"TOUCHPAY_{country}_"
        return {
            "base_url": base_url,
            "username": os.getenv(prefix + "USERNAME"),
            "password": os.getenv(prefix + "PASSWORD"),
            "partner_id": os.getenv(prefix + "PARTNER_ID"),
            "agence": os.getenv(prefix + "AGENCE"),
            "login_agent": os.getenv(prefix + "LOGIN_AGENT"),
            "password_agent": os.getenv(prefix + "PASSWORD_AGENT"),
            "service_code": payment_method.service_code or payment_method.code,
        }

    # --- 2) Campay (Collect / Withdraw) ---
    elif code.startswith("CAMPAY"):
        prefix = f"CAMPAY_{country}_"
        return {
            "base_url":       os.getenv(prefix + "BASE_URL"),      # ex: https://demo.campay.net
            "perm_token":     os.getenv(prefix + "PERM_TOKEN"),    # jeton permanent
            "webhook_secret": os.getenv(prefix + "WEBHOOK_SECRET"),# pour valider la signature (optionnel)
            "service_code":   payment_method.service_code or "",
        }

    # --- Aucun provider trouvé ---
    else:
        raise ValueError(f"Aucun provider configuré pour le code `{payment_method.code}`")