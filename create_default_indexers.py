from __future__ import unicode_literals

from django.db import migrations


def forwards_func(apps, schema_editor):
    CustomIndexer = apps.get_model("dvaapp", "CustomIndexer")
    db_alias = schema_editor.connection.alias
    CustomIndexer.objects.using(db_alias).bulk_create([
        CustomIndexer(name="inception", code="us"),
        CustomIndexer(name="facenet", code="fr"),
    ])


def reverse_func(apps, schema_editor):
    # forwards_func() creates two Country instances,
    # so reverse_func() should delete them.
    Country = apps.get_model("dvaapp", "Country")
    db_alias = schema_editor.connection.alias
    Country.objects.using(db_alias).filter(name="USA", code="us").delete()
    Country.objects.using(db_alias).filter(name="France", code="fr").delete()


class Migration(migrations.Migration):

    dependencies = [
        ('dvaapp', '0008_auto_20170502_2120'),
    ]

    operations = [
        migrations.RunPython(forwards_func, reverse_func),
    ]
