from django.urls import path
from . import views


app_name = 'myids'
urlpatterns = [
    path('', views.view_index, name='index'),
    path('stat', views.view_stat, name='stat'),
    path('conn', views.view_conn, name='conn'),
    path('query_conn', views.query_conn, name='query_conn'),
    path('query_stat', views.query_stat, name='query_stat'),
]