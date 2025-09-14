from django.urls import path
from .api_views import PaysListAPIView, SousSystemeListAPIView

app_name = 'resources_api'
urlpatterns = [
    path('pays/',          PaysListAPIView.as_view(),         name='pays_list'),
    path('sous-systeme/',  SousSystemeListAPIView.as_view(), name='soussysteme_list'),
]


