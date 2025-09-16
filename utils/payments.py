import os


def get_touchpay_config_for_method(payment_method):
    """
    Récupère la config sensitive (dans .env) selon le payment_method.
    Version ORIGINALE qui fonctionnait.
    """
    country_code = payment_method.pays.code.upper() if payment_method.pays else "CMR"

    prefix = f"TOUCHPAY_{country_code}_"

    return {
        "username": os.getenv(prefix + "USERNAME"),
        "password": os.getenv(prefix + "PASSWORD"),
        "partner_id": os.getenv(prefix + "PARTNER_ID"),
        "agence": os.getenv(prefix + "AGENCE"),
        "login_agent": os.getenv(prefix + "LOGIN_AGENT"),
        "password_agent": os.getenv(prefix + "PASSWORD_AGENT"),
        "service_code": payment_method.service_code or payment_method.code,
    }