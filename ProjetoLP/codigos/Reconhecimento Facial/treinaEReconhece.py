# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 05:41:46 2019

@author: Felizmelo
"""

# -*- coding: cp1252 -*-

# PCA - Principal component analysis
# É uma técnica utilizada para comprimir informações sem perder a essência
# O PCA faz isso com matrizes utilizando autovalores e autovetores
# O AutoFaces compacta a imagem do rosto de uma maneira que fica fácil comparar

# Funciona assim:
# O ViolaJones é o programa responsável por detectar as faces no vídeo
# Uma vez detectadas, ele envia as fotos apenas do rosto para o AutoFaces
# O AutoFaces faz o reconhecimento do rosto utilizando aquela imagem enviada


import numpy as np
import cv2
import os


def criaArquivoDeRotulo(pasta):
    # cria o arquivo de treino, que é um .txt com várias linhas no formato:
    # 'caminho completo da imagem';'rótulo'
    # sendo label o que classifica a foto. É apenas um número inteiro,
    # para identificar que é a pessoa 1 ou pessoa 2
    label = 0
    f = open("TRAIN", "w+")
    for dirPrincipal, nomeDirs, nomeArqs in os.walk(pasta):
        for subDir in nomeDirs:
            caminhoPasta = os.path.join(dirPrincipal, subDir)
            for filename in os.listdir(caminhoPasta):
                caminhoAbs = caminhoPasta + "\\" + filename
                f.write(caminhoAbs + ";" + str(label) + "\n")
            label = label + 1
    f.close()


def criaDicionarioDeImagens(fPoint):
    # Abre o arquivo de treino e cria um dicionário de fotos (já com as fotos abertas):
    # dicionarioDeFotos = {
    # 0: [Imagem1, Imagem2, ...],
    # 1: [Imagem1, Imagem2, ...],
    # ... }
    # A chave é o número do rótulo e o conteúdo é uma lista de fotos
    lines = fPoint.readlines()

    dicionarioDeFotos = {}
    for line in lines:
        filename, label = line.rstrip().split(';')
        if int(label) in dicionarioDeFotos.keys():
            dicionarioDeFotos.get(int(label)).append(cv2.imread(filename, 0))
        else:
            dicionarioDeFotos[int(label)] = [cv2.imread(filename, 0)]

    # ao final, cria um dicionário que na posição 0 (rótulo
    # referente a camila, por exemplo, tem-se uma lista contendo
    # todas as fotos da camila que estão na base de teste

    # na posição 1 do dicionário, teremos uma lista com todas as
    # fotos do pirula
    return dicionarioDeFotos


def treinaModelo(dicionarioAlunos):
    # cria e treina as autofaces
    model = cv2.face.EigenFaceRecognizer_create()
    listkey = []
    listvalue = []
    for key in dicionarioAlunos.keys():
        for value in dicionarioAlunos[key]:
            listkey.append(key)
            listvalue.append(value)

    model.train(np.array(listvalue), np.array(listkey))
    # param 1: todas as imagens:
    # [Imagem1, Imagem2, Imagem3, Imagem4, Imagem5, ...]
    # param 2: lista com todos os rótulos, na mesma ordem e com o mesmo tamanho do vetor das fotos:
    # [0, 0, 1, 0, 1, ...]
    return model


def reconheceVideo(modelo, arquivo):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(arquivo)  # inicia captura da câmera
    counterFrames = 0;
    while (counterFrames < 30):  # quando chegar ao milésimo frame, para
        ret, img = cap.read()

        # frame não pode ser obtido? entao sair
        if (ret == False):
            cap.release()
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # se nenhuma face for achada, continue
        if not np.any(faces):
            continue

        rostos = []
        # achou uma face? recorte ela (crop)
        for (x, y, w, h) in faces:
            rosto = img[y:y + h, x:x + w]
            # esse rosto é grande o bastante pra ser levado
            # em conta
            if w > 100 and h > 100:

                # modifica o tamanho dele pra se ajustar ao
                # treinamento e pinte pra tons de cinza
                rosto = cv2.resize(rosto, (255, 255))
                rosto = cv2.cvtColor(rosto, cv2.COLOR_BGR2GRAY)

                # aqui ele recebe a foto e diz qual rótulo
                # pertence (ou seja, quem é)
                label = modelo.predict(rosto)

                font = cv2.FONT_HERSHEY_SIMPLEX
                if (label[0] == 0):  # é o Felizmelo ?
                    # então bota um texto em cima da caixinha
                    #cv2.putText(img, 'Felizmelo', (x - 20, y + h + 60), font, 3, (255, 0, 0), 5, cv2.LINE_AA)
                    # (imagem, texto, posicao, fonte, tamanho da  fonte, cor, expessura, antialiasing)
                    # pinte um retângulo ao redor do rosto do Felizmelo 
                    img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                if (label[0] == 1):  # é Marciano?
                    # então bota um texto em cima da caixinha
                   # cv2.putText(img, 'Marciano', (x - 20, y + h + 60), font, 3, (0, 0, 255), 5, cv2.LINE_AA)
                    # pinte um retângulo ao redor do rosto de  Marciano
                    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # redimensione só pra ficar bonito na tela
        img = cv2.resize(img, (int(0.75 * img.shape[1]), int(0.75 * img.shape[0])))

        # exibir na tela!
        cv2.imshow("reconhecimento", img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    # cria um arquivo que indica que aquela foto pertence a tal pessoa
    criaArquivoDeRotulo("data")

    # carrega o arquivo
    fPoint = open("TRAIN", "r")

    # constrói um dicionário dos dados lidos no texto
    dicionarioDeFotos = criaDicionarioDeImagens(fPoint)
    modelo = treinaModelo(dicionarioDeFotos)

    # DO THE F* MAGIC DUDE
    reconheceVideo(modelo, "Felizmelo_Marciano.mp4")


if __name__ == "__main__":
    main()
