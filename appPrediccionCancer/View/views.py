from appPrediccionCancer.Logica import modeloSNN
from rest_framework.decorators import api_view
#from django.views.decorators.csrf import csrf_exempt0
#from django.http import JsonResponse
from django.shortcuts import render
import cv2
import numpy as np
import base64


class Clasificacion():
    def determinarCancer(request):
        return render(request, "cargadoImagen.html")
    @api_view(['GET','POST'])
    def predecir(request):
        try:
            imagen = request.FILES.get('imagen')

            if imagen:
                img_array = np.frombuffer(imagen.read(), np.uint8)
                imagen_cv2 = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                print("Dimensiones de la imagen:", imagen_cv2.shape)
                print("Tipo de datos de la imagen:", imagen_cv2.dtype)
                resul = modeloSNN.modeloSNN.predecir(modeloSNN.modeloSNN, imagen_cv2)

                escala = 4
                width = int(imagen_cv2.shape[1] * escala / 100)
                height = int(imagen_cv2.shape[0] * escala / 100)
                imagen_cv2_resized = cv2.resize(imagen_cv2, (width, height))

                imagen_rgb = cv2.cvtColor(imagen_cv2_resized, cv2.COLOR_BGR2RGB)
                imagen_jpg = cv2.imencode('.jpg', imagen_rgb)[1].tobytes()
                imagen_codificada = base64.b64encode(imagen_jpg).decode('utf-8')
            else:
                resul = 'No se ha seleccionado ninguna imagen'
                imagen_codificada = ''
        except:
            resul = 'Error al procesar la imagen'
            imagen_codificada = ''
        return render(request, "cargadoImagen.html", {"e": resul, "imagen_codificada": imagen_codificada})    
    """@csrf_exempt
    @api_view(['GET','POST'])
    def predecirIOJson(request):
        try:
            imagen = request.FILES.get('imagen')

            if imagen:
                img_array = np.frombuffer(imagen.read(), np.uint8)
                imagen_cv2 = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                print("Dimensiones de la imagen:", imagen_cv2.shape)
                print("Tipo de datos de la imagen:", imagen_cv2.dtype)
                resul = modeloSNN.modeloSNN.predecir(modeloSNN.modeloSNN, imagen_cv2)
            else:
                resul = 'No se ha seleccionado ninguna imagen'

            return JsonResponse({'e': resul}) 

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)"""