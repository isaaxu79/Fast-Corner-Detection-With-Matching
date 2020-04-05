from tkinter import *
from tkinter import ttk,filedialog,StringVar
from PIL import ImageTk, Image
from matplotlib import pyplot as plt
import numpy as np
import cv2
import random
from matplotlib.patches import ConnectionPatch
from decimal import Decimal
import threading
import imutils

fast = cv2.FastFeatureDetector_create(threshold=38) #38 para escala
root = Tk()
root.title("Fast Corner Detection")
root.geometry("500x300")
root.configure(background='#EFF0F1')
root.resizable(False,False)
variable = StringVar()
var = StringVar()
image_original = None
image_test = None
modi=[]
kpbst = 0
the_original= None

def analisis(key1, keyorigin,im1,im2):
    print("oorif: ", len(key1))
    key2 = fast.detect(im2,None)
    print("fast k2",len(key2))
    print("len k2",len(key2))
    asigna = asign(key2)
    matvh = to_match(key1,keyorigin,asigna)
    return matvh

def asign(kp2):
    keypointsTrans =[]
    for i in kp2:
        x = int(i.pt[0])
        y = int(i.pt[1])
        keypointsTrans.append([x,y,True])
    return keypointsTrans

def initial():
    global image_original, image_test
    root.filename =  filedialog.askopenfilename(initialdir = "/Users/PC1/Pictures/",title = "Select file",filetypes = (("Photo files","*.jpg"),("all files","*.*")))
    imgURL= root.filename
    source_entry.config(state=NORMAL)
    source_entry.insert(0,imgURL)
    source_entry.config(state=DISABLED)
    print(imgURL)
    s = show_select_image(imgURL)
    image_original = cv2.imread(imgURL)
    image_original=cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
    image_test = cv2.imread(imgURL)
    image_test=cv2.cvtColor(image_test, cv2.COLOR_BGR2RGB)
    start_detect.configure(state=NORMAL)

def show_select_image(path):
    img = Image.open(path)
    img = img.resize((230, 230), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.place(x=40,y=50)
    panel.image = img
    return img

def mensaje():
    print(variable.get())
    data = variable.get()
    option= Decimal(data)
    if(len(data)>0):
        if(option == 1):
            titolo.config(text="Introduce los grados de la rotacion")
        if(option == 2):
            titolo.config(text="Introduce x,y de desplazamiento, \nejemplo (22,22)")
        if(option == 3):
            titolo.config(text="Introduce la escala de 20-100")

def getkeypoints(imagen_original):
    global kpbst
    kp1x = fast.detect(imagen_original,50)
    keylen = len(kp1x)
    kpbst= keylen
    return [kp1x,keylen]

def to_match(keypts,keyptsOriTransf,keyptsTranform):
    matchpts = []
    for coord in range(len(keyptsOriTransf)):
        x = keyptsOriTransf[coord][0]
        y = keyptsOriTransf[coord][1]
        keys = []
        iter_key = []
        senal = False
        for s in range(len(keyptsTranform)):
            a = keyptsTranform[s][0]
            b = keyptsTranform[s][1]
            pointx = np.abs(x-a)
            pointy = np.abs(y-b)
            
            #validate error range +-2px of the iteration with condition
            if pointx <= 2 and pointx >= 0 and pointy <= 2 and pointy >= 0 and keyptsTranform[s][2] == True and keyptsOriTransf[coord][2] == True:
                print(pointy)
                keys.append([pointx+pointy])
                iter_key.append([coord,s])
                senal = True
        if senal:
            punto_minimos = np.argmin(np.array(keys))
            matchpts.append([[int(keypts[iter_key[punto_minimos][0]].pt[0]), int(keypts[iter_key[punto_minimos][0]].pt[1])], [keyptsTranform[iter_key[punto_minimos][1]][0],keyptsTranform[iter_key[punto_minimos][1]][1]]])
            #print([[int(keyptsOriTransf[iter_key[punto_minimos][0]][0]), int(keyptsOriTransf[iter_key[punto_minimos][0]][1])], [keyptsTranform[iter_key[punto_minimos][1]][0],keyptsTranform[iter_key[punto_minimos][1]][1]]])
            keyptsTranform[iter_key[punto_minimos][1]][2] = False
            keyptsOriTransf[iter_key[punto_minimos][0]][2] = False
    return matchpts

# -------------------rotacion----------------------
def main_rotation(angle,data):
    imagen2 = rotar_imagen_original(angle,image_original)
    kpts2 = get_keypoints_by_original_rotation(image_original,imagen2,data[0],angle)
    matches = analisis(data[0],kpts2,image_original,imagen2)
    return [matches,imagen2,angle]

def rotar_imagen_original(angulo,imagen):
    dst = imutils.rotate_bound(imagen, angulo)
    return dst

def graph_rotation(modi):
    global kpbst
    ti = []
    ti.append("original")
    for s in modi:
        ti.append(s[2])
    da = []
    da.append((kpbst*100)/kpbst)
    for x in modi:
        da.append((len(x[0])*100)/kpbst)
    fig = plt.figure(u'Gráfica de barras') # Figure
    ax = fig.add_subplot(111) # Axes
    xx = range(len(da))
    ax.bar(xx, da, width=0.8, align='center')
    ax.set_xticks(xx)
    ax.set_xticklabels(ti,rotation="vertical")
    ax.set_ylabel("porcentaje de keypoints")
    plt.show()

def get_keypoints_by_original_rotation(img1, img2, keypoints, angulo):
    (heigt,width) = img1.shape[:2]
    (heigt1,width1)= img2.shape[:2]

    centro = (width1/2,heigt1/2) # encientra el centro
    dify = (heigt1-heigt)/2  #la diferencia de las alturas entre 2
    difx = (width1-width)/2  # la diferencia del ancho entre 2

    keysOri = []
    theta = np.radians(angulo)

    for i in keypoints:
        x = round((i.pt[0] + difx),3)
        y = round((i.pt[1] + dify),3)

        xi = centro[0] + np.cos(theta) * (x-centro[0]) - np.sin(theta) * (y- centro[1])
        yi = centro[1] + np.sin(theta) * (x-centro[0]) + np.cos(theta) * (y- centro[1])
        keysOri.append([int(xi), int(yi), True])
    return keysOri

# ---------------final rotacion-------------

#--------------------Traslacion-----------------------------
def main_traslacion(x,y,data):
    global image_original
    imagen2 = translate_original(image_original,x,y)
    kpts2 = get_keypoints_by_original_traslation(data[0],x,y)
    matches = analisis(data[0],kpts2,image_original,imagen2)
    return [matches,imagen2,[x,y]]

def get_keypoints_by_original_traslation(keypts,x,y):
    new_keys = []
    tx = x
    ty= y
    if(x<0):
        tx=0
    if(y<0):
        ty=0
    for key in keypts:
        x = int(key.pt[0]) + tx
        y = int(key.pt[1]) + ty
        new_keys.append([x,y,True])
    return new_keys


def translate_original(img, x,y):
    if(x<0):
        if(y<0): # [-1,-1]
            dst= cv2.copyMakeBorder(img,0,(y*-1),0,(x*-1),cv2.BORDER_CONSTANT,value=(255,0,0))
        if(y==0): # [-1, 0]
            dst= cv2.copyMakeBorder(img,0,0,0,(x*-1),cv2.BORDER_CONSTANT,value=(255,0,0))
        if(y>0): # [-1,1]
            dst= cv2.copyMakeBorder(img,y,0,0,(x*-1),cv2.BORDER_CONSTANT,value=(255,0,0))
    if(x==0):
        if(y>0): # [0,1]
            dst= cv2.copyMakeBorder(img,y,0,0,0,cv2.BORDER_CONSTANT,value=(255,0,0))
        if(y<0): # [0,-1]
            dst= cv2.copyMakeBorder(img,0,(y*-1),0,0,cv2.BORDER_CONSTANT,value=(255,0,0))
    if(x>0):
        if(y<0): # [1,-1]
            dst= cv2.copyMakeBorder(img,0,(y*-1),x,0,cv2.BORDER_CONSTANT,value=(255,0,0))
        if(y==0): # [1, 0]
            dst= cv2.copyMakeBorder(img,0,0,x,0,cv2.BORDER_CONSTANT,value=(255,0,0))
        if(y>0): # [1,1]
            dst= cv2.copyMakeBorder(img,y,0,x,0,cv2.BORDER_CONSTANT,value=(255,0,0))
    return dst

def graph_traslation(modi):
    global kpbst
    ti = []
    ti.append("original")
    for s in modi:
        ti.append(str(s[2][0])+','+str(s[2][1]))
    da = []
    da.append((kpbst*100)/kpbst)
    for x in modi:
        da.append((len(x[0])*100)/kpbst)
    fig = plt.figure(u'Gráfica de barras') # Figure
    ax = fig.add_subplot(111) # Axes
    xx = range(len(da))
    ax.bar(xx, da, width=0.8, align='center')
    ax.set_xticks(xx)
    ax.set_xticklabels(ti,rotation="vertical")
    ax.set_ylabel("porcentaje de keypoints")
    plt.show()

#---------------------final traslacion------------------------

#----------------------Escalas-------------------------------
def main_escalas(escal,data):
    imagen2 = scall_original(image_original,escal)
    kpts2 = get_keypoints_by_original_scala(data[0],escal)
    matches = analisis(data[0],kpts2,image_original,imagen2)
    return [matches,imagen2,escal]

def scall_original(img,ratio):
    newimg = img
    width = int(newimg.shape[1] * ratio / 100)
    height = int(newimg.shape[0] * ratio / 100)
    dsize = (width, height)
    output = cv2.resize(newimg, dsize)
    return output

def get_keypoints_by_original_scala(keypts,esca):
    kyps = []
    for k in keypts:
        x = int(k.pt[0])*(esca/100)
        y = int(k.pt[1])*(esca/100)
        kyps.append([int(x),int(y),True])
    return kyps

def graph_scala(modi):
    global kpbst
    ti = []
    ti.append("original")
    for s in modi:
        ti.append(s[2])
    da = []
    da.append((kpbst*100)/kpbst)
    for x in modi:
        da.append((len(x[0])*100)/kpbst)
    fig = plt.figure(u'Gráfica de barras') # Figure
    ax = fig.add_subplot(111) # Axes
    xx = range(len(da))
    ax.bar(xx, da, width=0.8, align='center')
    ax.set_xticks(xx)
    ax.set_xticklabels(ti,rotation="vertical")
    ax.set_ylabel("porcentaje de keypoints")
    plt.show()

#---------------------fin escalas------------------------------

def draw(im1,im2,match):
    fig  = plt.figure()
    aux1 = plt.subplot(121)
    aux2 = plt.subplot(122)
    
    aux1.imshow(cv2.cvtColor(im1,cv2.COLOR_BGR2RGB))
    aux2.imshow(cv2.cvtColor(im2,cv2.COLOR_BGR2RGB))

    for i in range(len(match)):
        xyA = (match[i][0])
        xyB = (match[i][1])
        color= (random.uniform(0,1),random.uniform(0,1),random.uniform(0,1))
        con = ConnectionPatch(xyA=xyB,xyB=xyA,coordsA="data",coordsB="data",axesA=aux2,axesB=aux1,color=color)
        aux1.plot(xyA[0],xyA[1],'ro',color=color,markersize=3)
        aux2.plot(xyB[0],xyB[1],'ro',color=color,markersize=3)
        aux2.add_artist(con)
    plt.show()

def iniciar():
    global the_original
    global image_original, image_test, titolo, modi
    data = variable.get()
    modi = []
    option= int(data)
    print(option)
    if(len(data)>0):
        if(option == 1):
            idx = 0
            best = None
            grade = option_entry.get()
            grade = int(grade)
            ap=grade
            dato =round(360/grade)
            data = getkeypoints(image_original)
            print("len de fast ",data[1])
            for x in range(dato-1):
                modi.append(main_rotation(ap,data))
                print(ap)
                ap+=(grade)
            for s in modi:
                tam = len(s[0])
                if(tam > idx):
                    idx = tam
                    best = s
            draw(image_original,best[1],best[0])
            graph_rotation(modi)
        if option == 2:
            idx = 0
            trs = option_entry.get()
            dara = []
            trs = int(trs)
            data = getkeypoints(image_original)
            print("len de fast ",data[1])
            puntos = [[0,trs],[trs,0],[0,-trs],[-trs,0], [-trs,trs],[trs,-trs],[trs,trs],[-trs,-trs]]
            for ds in puntos:
                modi.append(main_traslacion(ds[0],ds[1],data))
            for s in modi:
                tam = len(s[0])
                if(tam > idx):
                    idx = tam
                    best = s
            draw(image_original,best[1],best[0])
            graph_traslation(modi)
        if(option == 3):
            data = getkeypoints(image_original)
            print("len de fast ",data[1])
            scalas=[25,50,200,400]
            for sc in scalas:
                aux = main_escalas(sc,data)
                modi.append(aux)
                draw(image_original,aux[1],aux[0])
            graph_scala(modi)


Label(root,text="Source:").place(x=33,y=20)
source_entry = Entry(root, width=50)
source_entry.place(x=90,y=20)
source_entry.config(state=DISABLED)
upload_button = Button(root, text ="Upload",borderwidth= 0, command = initial)
upload_button.place(x=400,y=12)
upload_button.configure(background="#5CB85C",height=2,width=8)
radio_translate = Radiobutton(text="Transladar", variable=variable, value=2,command = mensaje)
radio_rotate = Radiobutton(text="Rotar", variable=variable, value=1,command = mensaje)
radio_scale = Radiobutton(text="Escalar", variable=variable, value=3,command = mensaje)
radio_rotate.place(x=290,y=120)
radio_translate.place(x=290,y=150)
radio_scale.place(x=290,y=180)
option_entry = Entry(root,width=30)
option_entry.place(x=290,y=90)
titolo = Label(root,text="")
titolo.place(x=290,y=50)
start_detect= Button(root, text ="Start Detect",borderwidth= 0, command = iniciar)
start_detect.configure(background="#5CB85C",height=2,width=12,state=DISABLED)
start_detect.place(x=330,y=220)

if __name__ == "__main__":
    root.mainloop()