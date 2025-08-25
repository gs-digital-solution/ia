from django.urls import path
from .api_views import UserRegisterAPIView
from .api_views import UserLoginAPIView
from .api_views import CreateDeviceMigrationRequestAPIView
from resources.api_views import PaysListAPIView, SousSystemeListAPIView

urlpatterns = [
    path(
        'register/',
        UserRegisterAPIView.as_view(),
        name='api_register'),
    path(
        'login/',
        UserLoginAPIView.as_view(),
        name='api_login'),
    path(
    'device-migration/',
    CreateDeviceMigrationRequestAPIView.as_view(),
    name='api_device_migration'),

    path(
        'pays/',
        PaysListAPIView.as_view(),
        name='api_pays_list'),
    path(
        'sous-systeme/',
        SousSystemeListAPIView.as_view(),
        name='api_soussysteme_list'),
]