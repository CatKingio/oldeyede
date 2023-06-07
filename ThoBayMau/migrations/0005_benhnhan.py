# Generated by Django 4.2 on 2023-04-15 14:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ThoBayMau', '0004_alter_bacsi_bacsi_avatar'),
    ]

    operations = [
        migrations.CreateModel(
            name='benhnhan',
            fields=[
                ('benhnhan_id', models.AutoField(primary_key=True, serialize=False)),
                ('benhnhan_name', models.CharField(max_length=100)),
                ('benhnhan_age', models.IntegerField(null=True)),
                ('benhnhan_address', models.CharField(max_length=150, null=True)),
                ('benhnhan_sdt', models.IntegerField(null=True)),
                ('benhnhan_email', models.CharField(max_length=2500, null=True)),
                ('benhnhan_avatar', models.ImageField(default=None, null=True, upload_to='images_benhnhan')),
            ],
        ),
    ]