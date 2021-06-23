from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='index'),
    path('api/creates-AI-art', views.create_image, name='generate_image'),
    
]
