_PROVIDERS = {
    'TOUCHPAY': 'paiement.providers.touchpay.TouchpayProvider',
    'CAMPAY': 'paiement.providers.campay.CampayProvider',
    'EXTERNE': 'paiement.providers.externe.ExterneProvider',  # NOUVEAU
    'LYGOS': 'paiement.providers.externe.ExterneProvider',  # Utilise le même provider
    'IKEEPAY': 'paiement.providers.ikeepay.IkeePayProvider',
}


def get_provider_for_method(payment_method):
    # Si c'est un paiement externe, utiliser ExterneProvider
    if payment_method.est_externe():
        from paiement.providers.externe import ExterneProvider
        return ExterneProvider(payment_method)

    # Sinon, logique normale:
    prefix = payment_method.code.split('_')[0].upper()
    path = _PROVIDERS.get(prefix)
    if not path:
        raise ValueError(f"Aucun provider pour {prefix}")
    module_path, cls_name = path.rsplit('.', 1)
    module = __import__(module_path, fromlist=[cls_name])
    return getattr(module, cls_name)(payment_method)