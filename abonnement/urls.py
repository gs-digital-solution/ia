from django.urls import path
from .views import activer_abonnement_gratuit

app_name = 'abonnement'
urlpatterns = [
    path(
        'activation-code/',
        activer_abonnement_gratuit,
        name='activation_code'),
]