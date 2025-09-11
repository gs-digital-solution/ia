from django.urls import path
from .api_views import (
    PaymentMethodListAPI,
    StartPaymentAPI,
    PaymentStatusAPI
)

app_name = 'paiement_api'
urlpatterns = [
    path(
        'methods/',
        PaymentMethodListAPI.as_view(),
        name='methods'),
    path(
        'start/',
        StartPaymentAPI.as_view(),
        name='start'),
    path(
        'status/<str:transaction_id>/',
        PaymentStatusAPI.as_view(),
        name='status'),
]