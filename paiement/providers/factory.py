"""
Usine à providers de paiement.
On mappe un préfixe (agrégateur) à la classe correspondante.
"""

_PROVIDERS = {
    # clé = préfixe du code PaymentMethod (avant le premier ‘_’)
    'TOUCHPAY': 'paiement.providers.touchpay.TouchpayProvider',
    # 'CAMPAY'  : 'paiement.providers.campay.CampayProvider',
    # 'ORANGEPAY': 'paiement.providers.orangepay.OrangePayProvider',
    # etc.
}

def get_provider_for_method(payment_method):
    """
    Instancie le provider adéquat en fonction de payment_method.code.
    Ex : code='TOUCHPAY_MTN_CM' → clé='TOUCHPAY'.
    """
    # Extraction du préfixe
    prefix = payment_method.code.split('_', 1)[0].upper()
    path = _PROVIDERS.get(prefix)
    if not path:
        raise ValueError(f"Aucun provider configuré pour l’agrégateur '{prefix}'")
    module_path, class_name = path.rsplit('.', 1)
    module = __import__(module_path, fromlist=[class_name])
    provider_cls = getattr(module, class_name)
    return provider_cls(payment_method)