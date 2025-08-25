from django.urls import path
from .api_views import PaysListAPIView, SousSystemeListAPIView

urlpatterns = [
    path('pays/', PaysListAPIView.as_view(), name='api_pays_list'),
    path('sous-systeme/', SousSystemeListAPIView.as_view(), name='api_soussysteme_list'),
]