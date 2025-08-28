from django.urls import path
from .api_views import UserRegisterAPIView
from .api_views import UserLoginAPIView
from .api_views import CreateDeviceMigrationRequestAPIView
from resources.api_views import PaysListAPIView, SousSystemeListAPIView
from .api_views import PasswordResetAPI
from .api_views import ProfileAPIView
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from .api_views import (SoumissionExerciceAPIView, StatutSoumissionAPIView ,
                        ClassesListAPIView,MatieresListAPIView,
                        TypesExerciceListAPIView,DepartementsListAPIView,
                        LeconsListAPIView,DownloadCorrigeAPIView)

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

    path(
        'password-reset/',
        PasswordResetAPI.as_view(),
        name='api_password_reset'),

    path(
        'profile/',
        ProfileAPIView.as_view(),
        name='api_profile'),

    path(
        'token/',
        TokenObtainPairView.as_view(),
        name='token_obtain_pair'),
    path(
        'token/refresh/',
        TokenRefreshView.as_view(),
        name='token_refresh'),

    path(
        'soumission/',
        SoumissionExerciceAPIView.as_view(),
        name='api_soumission'),
    path(
        'soumission/<int:soumission_id>/status/',
        StatutSoumissionAPIView.as_view(),
        name='api_soumission_status'),

    path(
        'departements/',
        DepartementsListAPIView.as_view(),
        name='api_departements'),
    path(
        'classes/',
        ClassesListAPIView.as_view(),
        name='api_classes'),
    path(
        'matieres/',
        MatieresListAPIView.as_view(),
        name='api_matieres'),
    path(
        'types-exercice/',
         TypesExerciceListAPIView.as_view(),
         name='api_types_exercice'),
    path(
        'lecons/',
        LeconsListAPIView.as_view(),
        name='api_lecons'),

    path(
        'soumission/<int:soumission_id>/download/',
        DownloadCorrigeAPIView.as_view(),
        name='api_download_corrige'),
]