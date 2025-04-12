from django.urls import path
from .views import home, classify_image, inquiry

urlpatterns = [
    path('', home, name="home"),  # Homepage URL
    path('classify/', classify_image),
    path('inquiry/', inquiry),
]
