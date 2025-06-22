from rest_framework.views import APIView
from rest_framework.response import Response
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from .main_graph_builder import main_graph

# ===================== Global =====================
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
            output =main_graph.invoke(input_grapg)
            pdf_id = output.get("pdf_id")
            return Response({"message": "Process started", "status": "success", "pdf_id": pdf_id}, status=200)
        except Exception as e:
            return Response({"error": str(e)}, status=500)
