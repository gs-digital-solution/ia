from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from .models import (
    CustomUser,
    AppConfig,
    FeedbackCorrection,
    DeviceConnectionHistory,
    DeviceMigrationRequest,
    SoumissionIA,
    DemandeCorrection,
    CorrigePartiel,
   ContactWhatsApp,
)
from abonnement.models import PromoCode


# Inline pour afficher les codes promo liés à l'utilisateur
class PromoCodeInline(admin.TabularInline):
    model = PromoCode
    fk_name = 'proprietaire'  # champ ForeignKey vers CustomUser
    extra = 0  # pas de lignes vides à l'ajout
    readonly_fields = ('code', 'partage_count', 'date_creation')
    can_delete = False


@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    # On étend les fieldsets existants pour ajouter nos champs
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
                'role',
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
                'role',
            ),
        }),
    )

    # Affichage en liste : on regroupe tout ici (une seule fois)
    list_display = (
        'username',
        'first_name',
        'whatsapp_number',
        'gmail',
        'code_promo',
        'pays',
        'role',
        'is_staff',
        'is_superuser',
    )

    readonly_fields = ('code_promo',)

    # L'Inline pour voir les PromoCode de l'utilisateur
    inlines = [PromoCodeInline]

    # Optionnel : empêcher la modification de certains champs
    def get_readonly_fields(self, request, obj=None):
        ro = list(self.readonly_fields)
        if obj:  # en modification
            ro.append('username')  # par ex. empêcher de changer le username
        return ro


@admin.register(AppConfig)
class AppConfigAdmin(admin.ModelAdmin):
    list_display = ['pdf_enabled', 'correction_enabled']


@admin.register(FeedbackCorrection)
class FeedbackCorrectionAdmin(admin.ModelAdmin):
    list_display = ("correction", "user", "note", "created_at")
    list_filter = ("note", "created_at")
    search_fields = ("comment",)


@admin.register(DeviceConnectionHistory)
class DeviceConnectionHistoryAdmin(admin.ModelAdmin):
    list_display = ("user", "device_id", "connection_date", "successful")
    list_filter = ("user", "device_id", "successful")
    search_fields = ("user_username", "device_id")
    date_hierarchy = "connection_date"


@admin.register(DeviceMigrationRequest)
class DeviceMigrationRequestAdmin(admin.ModelAdmin):
    list_display = (
        "user",
        "previous_device_id",
        "new_device_id",
        "status",
        "request_date",
        "decision_date",
        "user_date_joined",
        "get_migration_count",
    )
    list_filter = ("status", "request_date")
    search_fields = ("user_first_name", "previous_device_id", "new_device_id")

    def get_migration_count(self, obj):
        return obj.user.migration_requests.count()

    get_migration_count.short_description = "Nb demandes migration"


@admin.register(SoumissionIA)
class SoumissionIAAdmin(admin.ModelAdmin):
    list_display    = ['user', 'demande', 'statut', 'progression', 'date_creation']
    list_filter     = ['statut', 'date_creation']
    readonly_fields = ['date_creation', 'date_maj']


@admin.register(DemandeCorrection)
class DemandeCorrectionAdmin(admin.ModelAdmin):
    list_display = ('user', 'date_soumission', 'fichier','exercices_data','nom_fichier')
    # si besoin, vous pouvez ajouter readonly_fields = ('fichier',) ou d'autres champs


@admin.register(CorrigePartiel)
class CorrigePartielAdmin(admin.ModelAdmin):
    list_display = ('soumission', 'titre_exercice', 'date_creation')





@admin.register(ContactWhatsApp)
class ContactWhatsAppAdmin(admin.ModelAdmin):
    list_display = ('lien_whatsapp_truncated', 'actif', 'date_creation', 'date_mise_a_jour')
    list_filter = ('actif', 'date_creation')
    search_fields = ('lien_whatsapp', 'message_accueil')
    readonly_fields = ('date_creation', 'date_mise_a_jour')

    # Afficher un seul formulaire pour éviter les doublons
    def has_add_permission(self, request):
        # Limiter à un seul enregistrement
        if ContactWhatsApp.objects.count() >= 1:
            return False
        return super().has_add_permission(request)

    def lien_whatsapp_truncated(self, obj):
        """Affiche le lien tronqué dans la liste admin"""
        if len(obj.lien_whatsapp) > 50:
            return f"{obj.lien_whatsapp[:50]}..."
        return obj.lien_whatsapp

    lien_whatsapp_truncated.short_description = "Lien WhatsApp"

    fieldsets = (
        (None, {
            'fields': ('lien_whatsapp', 'message_accueil', 'actif')
        }),
        ('Dates', {
            'fields': ('date_creation', 'date_mise_a_jour'),
            'classes': ('collapse',)
        }),
    )