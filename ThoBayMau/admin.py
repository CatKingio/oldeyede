from django.contrib import admin
from . import models
# Register your models here.
from .models import bacsi, benhnhan
class FileAnhAdmin(admin.ModelAdmin):
    list_display = ('images_id', 'images_name', 'images_level', 'images') # Tao cot cho bang du lieu
    search_fields = ['images_id', 'images_name', 'images_level'] # Tim kiem theo cot Name, CapdoBenh, Images filePath
    list_filter = ('images_id', 'images_name')
    

admin.site.register(models.FileAnh, FileAnhAdmin)
#admin.site.register(models.Upload)


class bacsiAdmin(admin.ModelAdmin):
    list_display = ("bacsi_id",
                    "bacsi_name",
                    "bacsi_age",
                    "bacsi_locate",
                    "bacsi_position",
                    "bacsi_content",
                    "bacsi_avatar")
    search_fields = ['bacsi_name']
    list_filter = ['bacsi_name', 'bacsi_position']

admin.site.register(bacsi, bacsiAdmin)


class benhnhanAdmin(admin.ModelAdmin):
    list_display = ("benhnhan_id",
                    "benhnhan_name",
                    "benhnhan_age",
                    "benhnhan_address",
                    "benhnhan_sdt",
                    "benhnhan_email",
                    "benhnhan_status",
                    "benhnhan_avatar")
    search_fields = ['benhnhan_name']
    list_filter = ['benhnhan_name', 'benhnhan_sdt']

admin.site.register(benhnhan, benhnhanAdmin)