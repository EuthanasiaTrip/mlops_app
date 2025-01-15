from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework import status

from . import modelmanager
from . import pgconnector as dbcon

# Create your views here.
@api_view(['POST'])
def model_evaluate(request):
    manager = modelmanager.ModelManager()
    data = request.data
    if not data:
        return Response('', status=status.HTTP_400_BAD_REQUEST)
    result = manager.evaluate(data['hasEmptyData'], data['data'])
    return Response(result)

@api_view(['GET'])
def model_train(request):
    manager = modelmanager.ModelManager()
    api_key = request.GET.get("key")
    dbCon = dbcon.DBConnector()
    if not api_key or not dbCon.validateKey(api_key):
        return Response('Forbidden', status=403)
    
    result = manager.train(return_report=True)
    from django.http import HttpResponse
    return HttpResponse(result)

@api_view(['POST'])
def append_data(request):
    dbCon = dbcon.DBConnector()
    body = request.data
    api_key = body["apiKey"]
    if not api_key or not dbCon.validateKey(api_key):
        return Response('Forbidden', status=403)

    inserted_ids = []
    for row in body["data"]:
        newId = dbCon.insert_new_data(row)
        inserted_ids.append(newId)

    return Response(inserted_ids)