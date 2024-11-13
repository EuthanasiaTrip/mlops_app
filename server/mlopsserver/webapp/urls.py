from django.urls import path

from . import views

urlpatterns = [
    path("api/evaluate", views.model_evaluate, name="evaluate"),
]