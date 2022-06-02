from django.urls import path
from django.conf.urls.static import static
from django.conf import settings
from . import views

app_name = 'mainapp'

urlpatterns = [
    path('', views.index, name='index'),
    path('information/', views.information, name='information'),
    path('result/', views.result, name='result'),
    path('evaluate/', views.evaluate, name='evaluate'),

]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)