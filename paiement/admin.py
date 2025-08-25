from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import PaymentTransaction, PaymentMethod

@admin.register(PaymentMethod)
class PaymentMethodAdmin(admin.ModelAdmin):
    list_display = ("nom_affiche", "operateur", "pays", "code", "ussd", "actif")
    list_filter = ("pays", "operateur", "actif")
    search_fields = ("code", "nom_affiche", "operateur")

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