import cv2
import pandas as pd
import numpy as np
from keras.models import load_model


class modeloSNN():
    """Clase Modelo Procesamiento y RNN"""
    def cargarNN(self, nombreArchivo):
        model = load_model(nombreArchivo+'.h5')
        print("Red Neuronal Cargada desde Archivo") 
        return model
    
    def predecir(self, imagen):
        dic = {0:'LungFCP-01-0001', 1:'LungFCP-01-0002', 2:'LungFCP-01-0003', 3:'LungFCP-01-0004', 4:'LungFCP-01-0005', 5:'LungFCP-01-0006'}
        imagen = cv2.resize(imagen, (64,64))
        imagen2 = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        pixeles = imagen2.flatten()
        fila = pd.Series(pixeles)
        dfnew = pd.DataFrame(fila)
        dfnew = dfnew.T
        x = dfnew.values
        x = x.reshape(1,64,64,3)
        modelo = self.cargarNN(self,'Recursos/CNNOptimizada')
        pred = modelo.predict(x)
        pred_labels = np.argmax(pred, axis=1)
        ClaseProbabilidad=np.argmax(pred)
        prob = pred.tolist()[0][ClaseProbabilidad]
        prob = str(round(prob*100, 4)) + '%'
        salida = { "clase": dic[int (pred_labels)], "probabilidad":prob}
        print(salida)
        return salida