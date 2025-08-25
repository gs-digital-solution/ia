from django.urls import path
from .views import start_payment, payment_callback

app_name = "paiement"
urlpatterns = [
    path(
        'start/',
        start_payment,
        name='start_payment'),
path(
    'callback/',
    payment_callback,
    name='callback'),

]