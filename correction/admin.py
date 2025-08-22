from django.contrib import admin
from .models import AppConfig
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser
from .models import FeedbackCorrection

@admin.register(AppConfig)
class AppConfigAdmin(admin.ModelAdmin):
    list_display = ['pdf_enabled','correction_enabled']

@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    fieldsets = UserAdmin.fieldsets + (
        ('Champs personnalisés', {
            'fields': (
                'whatsapp_number',
                'gmail',
                'code_promo',
                'pays',
                'sous_systeme',
                'secret_question',
                'secret_answer',
            )
        }),
    )
    add_fieldsets = UserAdmin.add_fieldsets + (
        ('Champs personnalisés', {
            'fields': (
                'whatsapp_number',
                'gmail',
                'pays',
                'sous_systeme',
                'secret_question',
                'secret_answer',
            ),
        }),
    )
    list_display = ('username', 'whatsapp_number', 'gmail', 'code_promo', 'first_name', 'is_staff')
    readonly_fields = ('code_promo',)  # Empêcher la modif du code promo dans l’admin

    fieldsets = UserAdmin.fieldsets + (
            ('Rôle et personnalisation', {'fields': ('role',)}),
        )
    add_fieldsets = UserAdmin.add_fieldsets + (
            ('Rôle et personnalisation', {'fields': ('role',)}),
        )
    list_display = ('username', 'role', 'is_staff', 'is_superuser')
        # On NE met pas "role" dans le formulaire d'inscription user COTE PUBLIC, seulement côté admin.

@admin.register(FeedbackCorrection)
class FeedbackCorrectionAdmin(admin.ModelAdmin):
    list_display = ("correction", "user", "note", "created_at")
    list_filter = ("note", "created_at")
    search_fields = ("comment",)