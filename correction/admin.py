from django.contrib import admin
from .models import AppConfig
@admin.register(AppConfig)
class AppConfigAdmin(admin.ModelAdmin):
    list_display = ['pdf_enabled']