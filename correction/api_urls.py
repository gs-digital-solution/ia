from django.urls import path, include
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from .api_views import AppConfigAPIView
from .api_views import (
    UserRegisterAPIView,
    UserLoginAPIView,
    CreateDeviceMigrationRequestAPIView,
    PasswordResetAPI,
    ProfileAPIView,
    SoumissionExerciceAPIView,
    StatutSoumissionAPIView,
    DepartementsListAPIView,
    ClassesListAPIView,
    MatieresListAPIView,
    TypesExerciceListAPIView,
    LeconsListAPIView,
    DownloadCorrigeAPIView,
    HistoriqueCorrectionsAPIView,
    FeedbackAPIView,
    PartagerCorrigeAPIView,
    DebugExtractionAPIView,
)

# On importe les routes d’abonnement/paiement/ressources ici
urlpatterns = [
    # Authentification & profil
    path('register/',    UserRegisterAPIView.as_view(),      name='api_register'),
    path('login/',       UserLoginAPIView.as_view(),         name='api_login'),
    path('device-migration/', CreateDeviceMigrationRequestAPIView.as_view(), name='api_device_migration'),
    path('password-reset/', PasswordResetAPI.as_view(),      name='api_password_reset'),
    path('profile/',      ProfileAPIView.as_view(),          name='api_profile'),
    path('token/',        TokenObtainPairView.as_view(),     name='token_obtain_pair'),
    path('token/refresh/',TokenRefreshView.as_view(),        name='token_refresh'),

    # Abonnement
    path('abonnement/', include(('abonnement.api_urls', 'abonnement_api')), name='abonnement_api'),

    # Paiement
    path('paiement/', include(('paiement.api_urls', 'paiement_api')), name='paiement_api'),

    # Ressources : pays et sous-systèmes
    path('pays/',         include(('resources.api_urls', 'resources_api')), name='resources_api'),
    # (resources.api_urls contient les routes 'pays/' et 'sous-systeme/')

    # Soumission d’exercice & statut
    path('soumission/',                    SoumissionExerciceAPIView.as_view(),      name='api_soumission'),
    path('soumission/<int:soumission_id>/status/', StatutSoumissionAPIView.as_view(), name='api_soumission_status'),

    # Listes protégées
    path('departements/',  DepartementsListAPIView.as_view(),   name='api_departements'),
    path('classes/',       ClassesListAPIView.as_view(),       name='api_classes'),
    path('matieres/',      MatieresListAPIView.as_view(),      name='api_matieres'),
    path('types-exercice/',TypesExerciceListAPIView.as_view(), name='api_types_exercice'),
    path('lecons/',        LeconsListAPIView.as_view(),        name='api_lecons'),

    # Corrigé / historique / feedback / partage PDF / debug
    path('soumission/<int:soumission_id>/download/', DownloadCorrigeAPIView.as_view(), name='api_download_corrige'),
    path('historique/',    HistoriqueCorrectionsAPIView.as_view(), name='api_historique'),
    path('feedback/<int:correction_id>/', FeedbackAPIView.as_view(), name='api_feedback'),
    path('partager/<int:soumission_id>/', PartagerCorrigeAPIView.as_view(), name='api_partager'),
    path('debug-extraction/', DebugExtractionAPIView.as_view(), name='api_debug_extraction'),

    # route de BLOCAGE PDF ou SOUMISSION coté backend
    path('app-config/', AppConfigAPIView.as_view(), name='api_app_config'),

]
