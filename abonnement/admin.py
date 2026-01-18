from django.contrib import admin
from django.db.models import Q
from .models import SubscriptionType, PromoCode, UserAbonnement, GlobalSubscriptionConfig
from correction.models import CustomUser  # IMPORT AJOUTÉ


class UserAbonnementAdmin(admin.ModelAdmin):
    """Admin personnalisé pour UserAbonnement avec recherche par numéro WhatsApp"""

    # Champs affichés dans la liste
    list_display = ('utilisateur', 'get_whatsapp_number', 'abonnement', 'date_debut', 'date_fin', 'exercice_restants',
                    'statut')
    list_filter = ('abonnement', 'statut', 'date_debut')

    # Barre de recherche pour trouver un utilisateur
    search_fields = (
        'utilisateur__username',  # Nom d'utilisateur
        'utilisateur__first_name',  # Prénom
        'utilisateur__last_name',  # Nom de famille
        'utilisateur__email',  # Email
        'utilisateur__whatsapp_number',  # Numéro WhatsApp
        'utilisateur__gmail',  # Gmail
        'abonnement__nom',  # Nom de l'abonnement
    )

    # Pour une recherche plus précise par numéro WhatsApp
    def get_search_results(self, request, queryset, search_term):
        queryset, use_distinct = super().get_search_results(request, queryset, search_term)

        if search_term:
            # Recherche aussi par numéro WhatsApp (avec nettoyage)
            phone_clean = search_term.strip()

            # Nettoyer le numéro pour la recherche
            # Retirer +237, 237, espaces, etc.
            if phone_clean.startswith('+237'):
                phone_clean = phone_clean[4:]  # Retire +237
            elif phone_clean.startswith('237'):
                phone_clean = phone_clean[3:]  # Retire 237

            # Retirer tous les caractères non numériques
            phone_clean = ''.join(filter(str.isdigit, phone_clean))

            if phone_clean:
                # Recherche le numéro nettoyé
                queryset |= self.model.objects.filter(
                    Q(utilisateur__whatsapp_number__icontains=phone_clean)
                )

        return queryset, use_distinct

    # Méthode pour afficher le numéro WhatsApp dans la liste
    def get_whatsapp_number(self, obj):
        return obj.utilisateur.whatsapp_number if obj.utilisateur.whatsapp_number else "-"

    get_whatsapp_number.short_description = "Numéro WhatsApp"
    get_whatsapp_number.admin_order_field = 'utilisateur__whatsapp_number'

    # Champs à afficher dans le formulaire d'édition
    fieldsets = (
        ('Utilisateur', {
            'fields': ('utilisateur', 'abonnement', 'code_promo_utilise')
        }),
        ('Détails de l\'abonnement', {
            'fields': ('date_debut', 'date_fin', 'exercice_restants', 'statut')
        }),
    )

    # Ajout d'un champ de recherche rapide dans le formulaire d'ajout
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