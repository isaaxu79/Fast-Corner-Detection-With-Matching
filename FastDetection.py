from tkinter import *
from tkinter import ttk,filedialog,StringVar
from PIL import ImageTk, Image
from matplotlib import pyplot as plt
import numpy as np
import cv2
from decimal import Decimal

fast = cv2.FastFeatureDetector_create(threshold=150)
root = Tk()
root.title("Fast Corner Detection")
root.geometry("500x300")
root.configure(background='#EFF0F1')
root.resizable(False,False)
variable = StringVar()
image_original = None
image_test = None

def get_keypoints(imagen):
    kp = fast.detect(imagen,50)
    return kp

def draw_keypoints(imagen,keypoints):
    imga = cv2.drawKeypoints(imagen, keypoints, None,color=(255,0,0))
    return imga

def scalling(img, ratio):
    newimg = img
    width = int(newimg.shape[1] * ratio / 100)
    height = int(newimg.shape[0] * ratio / 100)
    dsize = (width, height)
    output = cv2.resize(newimg, dsize)
    return output

def rotate(img, angle):
    newimg = img
    rows,cols, _ = newimg.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    dst = cv2.warpAffine(newimg,M,(cols,rows))

    return dst

def translate(img, point):
    newimg = img
    rows,cols, _ = newimg.shape
    M = np.float32([[1,0,point[0]],[0,1,point[1]]])
    dst = cv2.warpAffine(newimg,M,(cols,rows))
    return dst

def matching(ima1,ima2,k1,k2):
    orb = cv2.ORB_create()
    des1 = orb.compute(ima1,k1)
    des1 = des1[1]
    des2 = orb.compute(ima2,k2)
    des2 = des2[1]
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    img3 = cv2.drawMatches(ima1,k1,ima2,k2,matches ,None, flags=2)
    plt.imshow(img3),plt.show()


def initial():
    global image_original, image_test
    root.filename =  filedialog.askopenfilename(initialdir = "/Users/PC1/Pictures/",title = "Select file",filetypes = (("Photo files","*.jpg"),("all files","*.*")))
    imgURL= root.filename
    source_entry.config(state=NORMAL)
    source_entry.insert(0,imgURL)
    source_entry.config(state=DISABLED)
    print(imgURL)
    show_select_image(imgURL)
    image_original = cv2.imread(imgURL)
    image_test = cv2.imread(imgURL)
    start_detect.configure(state=NORMAL)

def show_select_image(path):
    img = Image.open(path)
    img = img.resize((230, 230), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.place(x=40,y=50)
    panel.image = img

def iniciar():
    global image_original, image_test
    data = variable.get()
    option= Decimal(data)
    print(option)
    if(len(data)>0):
        if(option == 1):
            grade = option_entry.get()
            grade = Decimal(grade)
            d = rotate(image_test,grade)
            keypoints_original = get_keypoints(image_original)
            keypoints_test = get_keypoints(d)
            draw_image_original = draw_keypoints(image_original,keypoints_original)
            draw_image_test = draw_keypoints(d,keypoints_test)
            matching(image_original,d,keypoints_original,keypoints_test)
        elif(option == 2):
            rt = option_entry.get()
            rt = Decimal(rt)
            d = translate(image_test,[rt,1])
            keypoints_original = get_keypoints(image_original)
            keypoints_test = get_keypoints(d)
            draw_image_original = draw_keypoints(image_original,keypoints_original)
            draw_image_test = draw_keypoints(d,keypoints_test)
            matching(image_original,d,keypoints_original,keypoints_test)
        elif(option == 3):
            sca = option_entry.get()
            sca = Decimal(sca)
            d = scalling(image_test,sca)
            keypoints_original = get_keypoints(image_original)
            keypoints_test = get_keypoints(d)
            draw_image_original = draw_keypoints(image_original,keypoints_original)
            draw_image_test = draw_keypoints(d,keypoints_test)
            matching(image_original,d,keypoints_original,keypoints_test)

source_text= Label(root,text="Source:").place(x=33,y=20)
source_entry = Entry(root, width=50)
source_entry.place(x=90,y=20)
source_entry.config(state=DISABLED)
upload_button = Button(root, text ="Upload",borderwidth= 0, command = initial)
upload_button.place(x=400,y=12)
upload_button.configure(background="#5CB85C",height=2,width=8)
radio_translate = Radiobutton(text="Transladar", variable=variable, value=2)
radio_rotate = Radiobutton(text="Rotar", variable=variable, value=1)
radio_scale = Radiobutton(text="Escalar", variable=variable, value=3)
radio_rotate.place(x=290,y=120)
radio_translate.place(x=290,y=150)
radio_scale.place(x=290,y=180)
option_entry = Entry(root,width=30)
option_entry.place(x=290,y=90)
start_detect= Button(root, text ="Start Detect",borderwidth= 0, command = iniciar)
start_detect.configure(background="#5CB85C",height=2,width=12,state=DISABLED)
start_detect.place(x=330,y=220)

root.mainloop()