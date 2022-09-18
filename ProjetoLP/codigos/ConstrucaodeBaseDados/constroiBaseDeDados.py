# -*- coding: cp1252 -*-
import numpy as np
import cv2
import os

def carregaNomesASeremLidos(txt):
    listaNomeAlunos = []
    pFile = open(txt, "r")
    for line in pFile:
        listaNomeAlunos.append(line.rstrip())
    return listaNomeAlunos

def criaPastaComNomes(listaNomes):
    for nome in listaNomes:
        try:
            print("criou " + nome)
            os.mkdir(nome)
        except OSError:
            print("N�o foi poss�vel criar o diret�rio ou o mesmo j� existe.")

def salvaFacesDetectadas(nome):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(nome + ".mp4") #inicia captura da c�mera

    counterFrames = 0;
    while(counterFrames < 30): #quando chegar ao mil�simo frame, para
        print(counterFrames)
        ret, img = cap.read()

        #frame n�o pode ser obtido? entao sair
        if(ret == False):
            cap.release()
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        #se nenhuma face for achada, continue
        if not np.any(faces):
            continue

        #achou uma face? recorte ela (crop)
        for (x, y, w, h) in faces:
            rostoImg = img[y:y+h, x:x+w]

        #imagens muito pequenas s�o desconsideradas
        larg, alt, _ = rostoImg.shape
        if(larg * alt <= 20 * 20):
            continue

        #salva imagem na pasta
        rostoImg = cv2.resize(rostoImg, (255, 255))
        cv2.imwrite(nome + "/" + str(counterFrames)+".png", rostoImg)
        counterFrames += 1
            
    cap.release()

#fun��o principal da aplica��o
def main():
    listaNomeAlunos = carregaNomesASeremLidos("input.txt")
    criaPastaComNomes(listaNomeAlunos)

    for nome in listaNomeAlunos:
        print("Analisando: " + nome)
        salvaFacesDetectadas(nome)


if __name__ == "__main__":
    main()
