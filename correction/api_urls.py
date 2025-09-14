from django.urls import path
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
from resources.api_views import PaysListAPIView, SousSystemeListAPIView
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

app_name = 'correction_api'
urlpatterns = [
    path('register/',    UserRegisterAPIView.as_view(),      name='api_register'),
    path('login/',       UserLoginAPIView.as_view(),         name='api_login'),
    path('device-migration/', CreateDeviceMigrationRequestAPIView.as_view(), name='api_device_migration'),

    # ← Versions publiques, sans authentification
    path('pays/',         PaysListAPIView.as_view(),         name='api_pays_list'),
    path('sous-systeme/', SousSystemeListAPIView.as_view(),  name='api_soussysteme_list'),

    path('password-reset/', PasswordResetAPI.as_view(),      name='api_password_reset'),
    path('profile/',        ProfileAPIView.as_view(),         name='api_profile'),

    path('token/',          TokenObtainPairView.as_view(),    name='token_obtain_pair'),
    path('token/refresh/',  TokenRefreshView.as_view(),       name='token_refresh'),

    path('soumission/',                     SoumissionExerciceAPIView.as_view(),      name='api_soumission'),
    path('soumission/<int:soumission_id>/status/', StatutSoumissionAPIView.as_view(), name='api_soumission_status'),

    path('departements/',  DepartementsListAPIView.as_view(),   name='api_departements'),
    path('classes/',       ClassesListAPIView.as_view(),        name='api_classes'),
    path('matieres/',      MatieresListAPIView.as_view(),       name='api_matieres'),
    path('types-exercice/',TypesExerciceListAPIView.as_view(),  name='api_types_exercice'),
    path('lecons/',        LeconsListAPIView.as_view(),         name='api_lecons'),

    path('soumission/<int:soumission_id>/download/', DownloadCorrigeAPIView.as_view(), name='api_download_corrige'),
    path('historique/',    HistoriqueCorrectionsAPIView.as_view(), name='api_historique'),
    path('feedback/<int:correction_id>/', FeedbackAPIView.as_view(), name='api_feedback'),
    path('partager/<int:soumission_id>/', PartagerCorrigeAPIView.as_view(), name='api_partager'),
    path('debug-extraction/', DebugExtractionAPIView.as_view(), name='api_debug_extraction'),
]