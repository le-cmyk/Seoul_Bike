from django.urls import path
from django.contrib import admin
from django.urls import include, path
from . import views

urlpatterns = [
    path('', views.index1, name='index'),
    path('apply_Model', views.index, name='index'),
    path('apply_Graph', views.indexGraph, name='index'),
    path('apply_GraphModel', views.indexGraphModel, name='index'),
]