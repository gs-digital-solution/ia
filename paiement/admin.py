from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import PaymentTransaction, PaymentMethod


@admin.register(PaymentMethod)
class PaymentMethodAdmin(admin.ModelAdmin):
    list_display = ("nom_affiche", "operateur", "pays", "type_paiement", "code", "actif", "priorite")
    list_filter = ("pays", "operateur", "type_paiement", "actif")
    search_fields = ("code", "nom_affiche", "operateur")
    list_editable = ("actif", "priorite")

    fieldsets = (
        ("Informations générales", {
            'fields': ('code', 'nom_affiche', 'operateur', 'pays', 'type_paiement', 'actif', 'priorite')
        }),
        ("Paiements internes", {
            'fields': ('ussd', 'service_code', 'extra_config'),
            'classes': ('collapse',)
        }),
        ("Paiements externes", {
            'fields': ('lien_externe', 'instructions_externes'),
            'classes': ('collapse',)
        }),
        ("Affichage", {
            'fields': ('logo_url', 'description')
        }),
    )

@admin.register(PaymentTransaction)
class PaymentTransactionAdmin(admin.ModelAdmin):
    list_display = (
        "user", "payment_method", "status", "amount",
        "transaction_id", "phone", "created"
    )
    list_filter = ("payment_method__pays", "payment_method__operateur", "status", "created")
    search_fields = ("transaction_id", "phone", "user__first_name", "user__whatsapp_number")
    readonly_fields = ("raw_response",)
    date_hierarchy = "created"