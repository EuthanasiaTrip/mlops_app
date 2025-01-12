from django.urls import path

from . import views

urlpatterns = [
    path("api/evaluate", views.model_evaluate, name="evaluate"),
    path("api/append", views.append_data, name="append"),
    path("api/train", views.model_train, name="train"),
]