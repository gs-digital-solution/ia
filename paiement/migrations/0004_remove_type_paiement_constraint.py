from django.db import migrations

class Migration(migrations.Migration):

    dependencies = [
        ('paiement', '0003_add_iframe_payment_type'),
    ]

    operations = [
        migrations.RunSQL(
            sql="ALTER TABLE paiement_paymentmethod DROP CONSTRAINT IF EXISTS check_type_paiement;",
            reverse_sql=migrations.RunSQL.noop,  # Pas de retour possible
        ),
    ]