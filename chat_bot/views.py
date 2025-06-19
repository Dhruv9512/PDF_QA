from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
import os, io, re, uuid, markdown
from typing import Annotated
from typing_extensions import TypedDict
from .main_graph_builder import main_graph

# ===================== Global =====================
pdf_id = None
collection_name = "pdf_documents"

# ===================== Django APIView =====================
@method_decorator(csrf_exempt, name='dispatch')
class pdf(APIView):
     def post(self, request):
        # Referal = request.data.get("Referal")
        QuePdf = request.data.get("QuePdf")
        input_grapg = {
            # "Referal": io.BytesIO(Referal),
            "QuePdf": QuePdf,
            "collection_name": collection_name,
            "Ans": [],
            "messages": []
        }
        try:
            main_graph.invoke(input_grapg)
            return Response({"message": "Process started", "status": "success", "pdf_id": pdf_id}, status=200)
        except Exception as e:
            return Response({"error": str(e)}, status=500)
