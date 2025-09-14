from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path, include
from django.contrib.auth.views import LogoutView

urlpatterns = [
    # Interface Web & Admin
    path('admin/',      admin.site.urls),
    path('resources/',  include('resources.urls')),
    path('ckeditor/',   include('ckeditor_uploader.urls')),
    path('correction/', include('correction.urls')),
    path('logout/',     LogoutView.as_view(next_page='correction:soumettre'), name='logout'),
    path('chaining/',   include('smart_selects.urls')),
    path('abonnement/', include('abonnement.urls')),
    path('paiement/',   include('paiement.urls')),

    # Toutes les API REST passent par correction.api_urls
    path('api/', include(('correction.api_urls', 'correction_api'), namespace='correction_api')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)