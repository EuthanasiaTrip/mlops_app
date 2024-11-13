from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework import status

from . import modelmanager

# Create your views here.
@api_view(['POST'])
def model_evaluate(request):
    manager = modelmanager.ModelManager()
    data = request.data
    if not data:
        return Response('', status=status.HTTP_400_BAD_REQUEST)
    result = manager.evaluate(data['hasEmptyData'], data['data'])
    return Response(result)
