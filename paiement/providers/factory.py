_PROVIDERS = {
    'TOUCHPAY': 'paiement.providers.touchpay.TouchpayProvider',
    # plus tard : 'CAMPAY': 'paiement.providers.campay.CampayProvider', ...
}

def get_provider_for_method(payment_method):
    prefix = payment_method.code.split('_')[0].upper()
    path = _PROVIDERS.get(prefix)
    if not path:
        raise ValueError(f"Aucun provider pour {prefix}")
    module_path, cls_name = path.rsplit('.', 1)
    module = __import__(module_path, fromlist=[cls_name])
    return getattr(module, cls_name)(payment_method)