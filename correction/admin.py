from django.contrib import admin
from .models import AppConfig
@admin.register(AppConfig)
class AppConfigAdmin(admin.ModelAdmin):
    list_display = ['pdf_enabled']

from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser

@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    # On peut personnaliser les champs visibles ici
    fieldsets = UserAdmin.fieldsets + (
        ('Informations personnelles', {
            'fields': ('whatsapp_number', 'pays', 'sous_systeme', 'secret_question', 'secret_answer',)
        }),
    )
    add_fieldsets = UserAdmin.add_fieldsets + (
        ('Informations personnelles', {
            'fields': ('whatsapp_number', 'pays', 'sous_systeme', 'secret_question', 'secret_answer',)
        }),
    )
    list_display = ('username', 'first_name', 'whatsapp_number', 'pays', 'sous_systeme', 'is_staff')
    search_fields = ('username', 'first_name', 'whatsapp_number')
    ordering = ('username',)