# C:\Users\#####\AppData\Local\Programs\Python\Python313>
# pip install opencv-contrib-python
# https://omes-va.com/deteccion-de-objetos-dnn-opencv-python/

import cv2

# Leemos el modelo y los pesos
# Arquitectura de modelos
arquitectura = "./arquitecturaPesos/MobileNetSSD_deploy.prototxt.txt"
# pesos
pesos = "./arquitecturaPesos/MobileNetSSD_deploy.caffemodel"
# clases
soluciones = {0: "background", 1: "aeroplane", 2: "bicycle",
              3: "bird", 4: "boat",
              5: "bottle", 6: "bus",
              7: "car", 8: "cat",
              9: "chair", 10: "cow",
              11: "diningtable", 12: "dog",
              13: "horse", 14: "motorbike",
              15: "person", 16: "pottedplant",
              17: "sheep", 18: "sofa",
              19: "train", 20: "tvmonitor"}

# cargamos el modelo
model = cv2.dnn.readNetFromCaffe(arquitectura, pesos)
print(model)

##########################################################
# Empezamos a cargar la imagen
# Si no encuentra la imagen da error AttributeError (Try)
try:
    imagen = cv2.imread("./test/imageSalon.jpg")
    # Ese guion hace para que se ignore el valor en el
    height, width, _ = imagen.shape
    image_resized = cv2.resize(imagen, (400, 400))

    # Un Blob es un tipo de contenedor para elementos: BINARIOS GRANDES
    # blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)
    # Cuanto le robas a cada color para darle un procesamiento concreto
    # swapRB=True En RGB intercambiar R y B para algun uso que necesites despues
    # https://pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
    blob = cv2.dnn.blobFromImage(
        image_resized, 0.007843, (400, 400), (127.5, 127.5, 127.5))
    # print("blob.shape: ", blob.shape)

    # Detectecciones y Predicciones

    # Configura un nuevo valor para la capa Blob
    model.setInput(blob)
    detecciones = model.forward()
    # forward computa la salida net

    # print(detecciones)

# Ya tenemos las detecciones
    for deteccion in detecciones[0][0]:
        # print(deteccion)

        if deteccion[2] > 0.45:
            label = soluciones[deteccion[1]]
            print("Etiqueta:", label)
            box = deteccion[3:7] * [width, height, width, height]
            x_start, y_start, x_end, y_end = int(
                box[0]), int(box[1]), int(box[2]), int(box[3])
            # Hacemos el marquito
            cv2.rectangle(imagen, (x_start, y_start),
                          (x_end, y_end), (0, 255, 0), 2)
            # Creo que el Conf es la nota de semejanza
            cv2.putText(imagen, "Conf: {:.2f}".format(
                deteccion[2] * 100), (x_start, y_start - 5), 1, 1.2, (255, 0, 0), 2)
            # Escribimos el nombre de lo que hemos encontrado
            cv2.putText(imagen, label, (x_start, y_start - 25),
                        1, 1.2, (255, 0, 0), 2)
    cv2.imshow("Image", imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except AttributeError:
    print('SEGURAMENTE TE HAYAS EQUIVOCADO EN LA RUTA')
