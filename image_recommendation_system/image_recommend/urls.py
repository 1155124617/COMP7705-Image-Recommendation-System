from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name='index'),
    path("recommend", views.recommend, name='recommend'),
    path("recommend_similar", views.recommend_similar, name='recommend_similar'),
    path("transfer_styles", views.transfer_styles, name="transfer_styles"),
]