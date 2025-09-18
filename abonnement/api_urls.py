from django.urls import path
from .api_views import (
    SubscriptionTypeListAPI,
    ActivatePromoAPI,
    CurrentSubscriptionAPI
)

app_name = 'abonnement_api'
urlpatterns = [
    path(
        'types/',
        SubscriptionTypeListAPI.as_view(),
        name='types'),
    path(
        'activate-promo/',
        ActivatePromoAPI.as_view(),
        name='activate_promo'),
    path(
        'current/',
        CurrentSubscriptionAPI.as_view(),
        name='current'),
]
