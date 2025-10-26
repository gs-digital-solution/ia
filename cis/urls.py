from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),

    # Mettre les routes d'abonnement et paiement EN PREMIER
    path('api/abonnement/', include('abonnement.api_urls')),
    path('api/paiement/', include('paiement.api_urls')),

    # Routes de correction
    path('api/', include('correction.api_urls')),

    # Routes des ressources (pays, sous-systèmes)
    path('api/', include('resources.api_urls')),

  # Routes des ressources dynamique (exercices corrigés, matière, leçons, classes)
    path('resources/', include('resources.urls')),

# Routes pour afficher le corrige web et envoyer le PDF de ce corrigé coté flutter
    path(
        'correction/',
        include(('correction.urls', 'correction'), namespace='correction')),

    path('chaining/', include('smart_selects.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)