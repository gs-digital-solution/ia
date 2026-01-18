from django.contrib import admin
from .models import SubscriptionType, PromoCode, UserAbonnement, GlobalSubscriptionConfig


class UserAbonnementAdmin(admin.ModelAdmin):
    """
    Admin pour la gestion des abonnements utilisateur
    Avec barre de recherche et autocomplétion
    """

    # 1. BARRE DE RECHERCHE dans la liste principale
    search_fields = [
        'utilisateur__whatsapp_number',  # Recherche par numéro WhatsApp
        'utilisateur__username',  # Par nom d'utilisateur
        'utilisateur__email',  # Par email
        'utilisateur__first_name',  # Par prénom
        'utilisateur__last_name',  # Par nom
        'abonnement__nom',  # Par type d'abonnement
    ]

    # 2. AUTOMATIC COMPLETE pour le formulaire d'ajout
    # Transforme le dropdown en champ avec recherche 🔍
    autocomplete_fields = ['utilisateur', 'code_promo_utilise']

    # 3. Affichage dans la liste
    list_display = (
        'utilisateur',
        'get_whatsapp_number',  # Affiche le numéro WhatsApp
        'abonnement',
        'date_debut',  # Affiche mais ne modifie pas
        'date_fin',
        'statut',
        'exercice_restants'
    )

    # 4. Méthode pour afficher le numéro WhatsApp
    def get_whatsapp_number(self, obj):
        return obj.utilisateur.whatsapp_number

    get_whatsapp_number.short_description = "WhatsApp"
    get_whatsapp_number.admin_order_field = 'utilisateur__whatsapp_number'

    # 5. Filtres
    list_filter = ('abonnement', 'statut', 'date_debut')

    # 6. Pagination
    list_per_page = 20

    # 7. Organisation du formulaire - ENLEVER date_debut car auto_now_add=True
    fieldsets = (
        ('Utilisateur', {
            'fields': ('utilisateur', 'abonnement', 'code_promo_utilise')
        }),
        ('Détails', {
            'fields': ('date_fin', 'exercice_restants', 'statut')
            # date_debut est automatique, ne pas l'inclure
        }),
    )

    # 8. Champs en lecture seule
    readonly_fields = ('date_debut',)


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
    search_fields = ('code', 'proprietaire__username', 'proprietaire__whatsapp_number')
    autocomplete_fields = ['proprietaire']

    # Affiche le numéro WhatsApp du propriétaire
    def get_whatsapp_number(self, obj):
        return obj.proprietaire.whatsapp_number

    get_whatsapp_number.short_description = "WhatsApp"
    get_whatsapp_number.admin_order_field = 'proprietaire__whatsapp_number'

    # date_creation est auto_now_add, donc en lecture seule
    readonly_fields = ('date_creation',)


class GlobalSubscriptionConfigAdmin(admin.ModelAdmin):
    """Admin pour la configuration globale (Singleton)"""
    list_display = ('nb_gratuit_par_utilisateur', 'options_gratuit_promo',
                    'options_reduction_promo', 'options_normal')

    # Empêche d'ajouter plusieurs configurations
    def has_add_permission(self, request):
        if self.model.objects.count() >= 1:
            return False
        return super().has_add_permission(request)


# Enregistrement avec les classes d'admin personnalisées
admin.site.register(SubscriptionType, SubscriptionTypeAdmin)
admin.site.register(PromoCode, PromoCodeAdmin)
admin.site.register(UserAbonnement, UserAbonnementAdmin)
admin.site.register(GlobalSubscriptionConfig, GlobalSubscriptionConfigAdmin)