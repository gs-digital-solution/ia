from django.apps import AppConfig


class AbonnementConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'abonnement'


class AbonnementConfig(AppConfig):
    name = 'abonnement'

    def ready(self):
        import abonnement.models  # ou abonnement.signals si tu apostas le signal dans un fichier séparé