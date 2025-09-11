"""
URL configuration for cis project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path, include
from django.contrib.auth.views import  LogoutView

urlpatterns = [
    path('admin/', admin.site.urls),
    path(
        'resources/',
        include('resources.urls')
    ),
    path(
        'ckeditor/',
        include('ckeditor_uploader.urls')),
path(
    'correction/',
    include('correction.urls')),

path(
    'logout/',
    LogoutView.as_view(next_page='correction:soumettre'),
    name='logout'),

path(
    'chaining/',
    include('smart_selects.urls')),

path(
    'abonnement/',
    include('abonnement.urls')),

path(
    'paiement/',
    include('paiement.urls')),

path(
    'api/',
    include('correction.api_urls')),

    path(
        'api/abonnement/',
        include(('abonnement.api_urls', 'abonnement_api'),
                                  namespace='abonnement_api')),

    path(
        'api/paiement/',
        include(('paiement.api_urls', 'paiement_api'),
                                  namespace='paiement_api')),

]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

