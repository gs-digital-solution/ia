from django.urls import path
from .api_views import UserRegisterAPIView
from .api_views import UserLoginAPIView
from .api_views import CreateDeviceMigrationRequestAPIView

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
]