from django.contrib import admin
from django.db.models import Q
from .models import SubscriptionType, PromoCode, UserAbonnement, GlobalSubscriptionConfig

# Utilisez get_user_model au lieu d'importer directement
from django.contrib.auth import get_user_model

User = get_user_model()


class UserAbonnementAdmin(admin.ModelAdmin):
    list_display = ('utilisateur', 'abonnement', 'date_debut', 'statut')
    list_filter = ('abonnement', 'statut')

    # Version SIMPLIFIÉE de search_fields
    search_fields = (
        'utilisateur__username',
        'utilisateur__email',
    )

    # Enlevez get_search_results temporairement
    # Enlevez get_whatsapp_number temporairement

    fieldsets = (
        ('Utilisateur', {
            'fields': ('utilisateur', 'abonnement', 'code_promo_utilise')
        }),
        ('Détails', {
            'fields': ('date_debut', 'date_fin', 'exercice_restants', 'statut')
        }),
    )

    raw_id_fields = ('utilisateur',)

    # Nombre d'éléments par page
    list_per_page = 20


class SubscriptionTypeAdmin(admin.ModelAdmin):
    """Admin pour les types d'abonnement"""
    list_display = ('nom', 'code', 'prix_base', 'duree_jours', 'nombre_exercices_total', 'actif')
    list_filter = ('actif', 'code')
    search_fields = ('nom', 'description')
    list_editable = ('actif',)


class PromoCodeAdmin(admin.ModelAdmin):
    """Admin pour les codes promo"""
    list_display = ('code', 'proprietaire', 'get_whatsapp_number', 'partage_count', 'actif', 'date_creation')
    list_filter = ('actif', 'date_creation')
    search_fields = ('code', 'proprietaire__username', 'proprietaire__email', 'proprietaire__whatsapp_number')
    raw_id_fields = ('proprietaire',)

    # Méthode pour afficher le numéro WhatsApp dans la liste
    def get_whatsapp_number(self, obj):
        return obj.proprietaire.whatsapp_number if obj.proprietaire.whatsapp_number else "-"

    get_whatsapp_number.short_description = "Numéro WhatsApp"
    get_whatsapp_number.admin_order_field = 'proprietaire__whatsapp_number'


class GlobalSubscriptionConfigAdmin(admin.ModelAdmin):
    """Admin pour la configuration globale (Singleton)"""
    list_display = ('nb_gratuit_par_utilisateur', 'options_gratuit_promo',
                    'options_reduction_promo', 'options_normal')


# Enregistrement avec les classes d'admin personnalisées
admin.site.register(SubscriptionType, SubscriptionTypeAdmin)
admin.site.register(PromoCode, PromoCodeAdmin)
admin.site.register(UserAbonnement, UserAbonnementAdmin)
admin.site.register(GlobalSubscriptionConfig, GlobalSubscriptionConfigAdmin)