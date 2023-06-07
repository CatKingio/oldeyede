from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings


urlpatterns = [
    path('', views.index),
    path('Dashboard/', views.dashboard, name="home"),
    path('Chat/', views.Chatting),
    path('ThongTinCaNhan/', views.Profile),
    path('Calendar/', views.Calendar),
    path('Doctors/', views.get_bacsi),
    path('Thembacsi/', views.AddDoctors),
    path('BenhNhan/', views.get_benhnhan),
    path('TinhTrangBenhNhan/', views.TinhHinhBenhNhan),
    path('Uploadfile/', views.Uploadfiles),
    path('PhanTich/', views.ThongKe, name='thongke'),
    path('Dudoan/', views.Dudoan, name='Dudoan'),
    path('Home/', views.index),
    path('ChinhSuaThongTin/', views.EditProfile),
    
    path('DangNhap/', views.login_user, name='login_user'),
    path('DangKy/', views.register, name='register'),
    path('predict/', views.predict, name='predict'),
]


urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
