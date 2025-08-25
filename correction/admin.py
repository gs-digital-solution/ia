from django.contrib import admin
from .models import AppConfig
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser
from .models import FeedbackCorrection
from .models import DeviceConnectionHistory, DeviceMigrationRequest

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


##*3. Admin Django – pour valider/voir/modifier les demandes et historique*
@admin.register(DeviceConnectionHistory)
class DeviceConnectionHistoryAdmin(admin.ModelAdmin):
    list_display = ("user", "device_id", "connection_date", "successful")
    list_filter = ("user", "device_id", "successful")
    search_fields = ("user__username", "device_id")
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
        "get_migration_count"
    )
    list_filter = ("status", "request_date")
    search_fields = ("user__first_name", "previous_device_id", "new_device_id")

    def get_migration_count(self, obj):
        return obj.user.migration_requests.count()
    get_migration_count.short_description = "Nb demandes migration"