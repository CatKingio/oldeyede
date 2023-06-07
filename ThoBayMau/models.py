from django.db import models
from django.utils import timezone

# Create your models here.


class FileAnh(models.Model):
    images_id = models.AutoField(primary_key=True)  # Dùng ID làm khóa chính
    # another_id = models.ForeignKey(bảng khác, default=None, on_delete=models.CASCADE)
    # null=True cho phep co the khong can nhap
    images_name = models.CharField(max_length=50, null=True)
    images_level = models.IntegerField(null=False)  # cap do benh
    images = models.ImageField(
        upload_to='images', null=True, blank=True, default=None)

    def __str__(self):
        return f"{self.images_id}, {self.images_name}, {self.images_level}, {self.images}"


class Upload(models.Model):
    imagespath = models.TextField()
    images = models.ImageField(null=True, blank=True)
    predicted = models.TextField()
    confidence = models.IntegerField(default=0, null=True, blank=True)
    # Save images bằng thời gian hiện tại gán cho id
    saved = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ('-saved',)

    def __str__(self):
        return self.imagespath


class bacsi(models.Model):
    bacsi_id = models.AutoField(primary_key=True)
    bacsi_name = models.CharField(max_length=150, null=False)
    bacsi_age = models.IntegerField(null=True)
    bacsi_locate = models.CharField(max_length=150, null=True)
    bacsi_position = models.CharField(max_length=150, null=True)
    bacsi_content = models.CharField(max_length=2500, null=True)
    bacsi_avatar = models.ImageField(
        upload_to='images_bacsi', null=True, default=None)

    def __str__(self):
        return f"{self.bacsi_id},{self.bacsi_name},{self.bacsi_age},{self.bacsi_locate},{self.bacsi_position},{self.bacsi_content},{self.bacsi_avatar}"


class benhnhan(models.Model):
    benhnhan_id = models.AutoField(primary_key=True)
    benhnhan_name = models.CharField(max_length=100, null=False)
    benhnhan_age = models.IntegerField(null=True)
    benhnhan_address = models.CharField(max_length=150, null=True)
    benhnhan_sdt = models.CharField(max_length=11,null=True)
    benhnhan_email = models.CharField(max_length=500, null=True)
    benhnhan_status = models.CharField(max_length=500, null=True)
    benhnhan_avatar = models.ImageField(upload_to='images_benhnhan', null=True, default=None)

    def __str__(self):
        return f"{self.benhnhan_id},{self.benhnhan_name},{self.benhnhan_age},{self.benhnhan_address},{self.benhnhan_sdt},{self.benhnhan_email},{self.benhnhan_status},{self.benhnhan_avatar}"
