from flask import Flask, render_template, request
import pandas as pd
from sklearn.externals import joblib
from PIL import Image
import numpy as np
import urllib.request
import urllib.parse

import keras
from keras.models import load_model
from keras.preprocessing import image as im

from skimage import color
from skimage import io
# img = color.rgb2gray(io.imread('image.png'))

import cv2

# -----serail----
# import tesseract
import pytesseract
# from google.colab.patches import cv2_imshow
# ---------------------------

import tensorflow as tf
global graph,model2,autoencoder
graph = tf.get_default_graph()

app = Flask(__name__)
model2 = load_model('final.h5')    #Lenet-aw.h5

autoencoder = load_model('happyHack.h5')

# model = keras.Sequential()
# model = load_model('LeNetArch.h5')
# model = load_model('litemodel.sav')
# model = joblib.load('lenet-aw.h5')   
# 'NoteDetection.h5'  'litemodel.sav'  LeNetArch.h5

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def get():
	input = dict(request.form)
	# accidentindex = input['accidentindex'][0]
	# age_of_vehicle = input['age_of_vehicle'][0]

	# if request.method == 'POST':  
	#     f = request.files['file']  
	#     f.save(f.filename) 

	# file = request.files['file']
	# if request.method == 'POST':  
	#     f = request.files['file']
	#     f.save(f.filename)  
	# filename= request.form['hidden']  #input['imgname']
	# filename = request.form.get('imgname')
	print("filename=",request.form.get('imgname'))  

	print("filename=",input['imgname'][0])   

	v= input['imgname'][0]
	#session
	# s = session['key']
	# print("s =",s) 
	 # --------------------------------------------------------------
	 # grey scale
	# img = color.rgb2gray(io.imread(v))
	# img = im.load_img(v, target_size=(256, 256))
	# img = color.rgb2gray(io.imread(img))

	# data = np.array([np.array(img)]) 
	# data = np.array([np.array(Image.open(v))])  #f.filename
	i=cv2.imread(v,0); # input as grey scale
	width = 256
	height = 256
	dim = (width, height)

	# resize image
	resized = cv2.resize(i, dim, interpolation = cv2.INTER_AREA)

	data = np.array([np.array(resized)])
	# -----------------------------------------------------


	# data = np.array([np.array(Image.open('500.jpg'))])  #/utilities/
	data = data.reshape(-1,256,256,1)
	data = data.astype('float32')
	data = data/ 255.
	# print("logging ",data)
	# return str(data);

	# model2._make_predict_function()

	with graph.as_default():
	 y = model2.predict(data, verbose=1)

	# try: y =model2.predict(data, verbose=1)
	# except Exception as e: y = str(e)

	# y =model2.predict(data, verbose=1)
	# y = y.astype('float32')
	# r = [round(x[0]) for x in y]
	# r = float(round(y[0][0]))
	# res = np.array(y, dtype='float64')
	# res = np.array(r, dtype='float64')
	print(" Y =  ",y)
	# print(" res =  ",res)
	# result = int(round(y[0][0])) 
	# https://thispointer.com/find-max-value-its-index-in-numpy-array-numpy-amax/
	result = np.where(y[0] == np.amax(y[0]))
	print("result[0]=",result[0])
	print("result[0][0]=",result[0][0])
	pre=result[0][0]
	# ------------------------
	# return str(result[0][0])

	# -------------------serial number -----------
	img2 = cv2.imread(v, 0)
	# cropped=img2[115:130, 30:100]
	if pre == 0:
	    cropped=img2[115:130, 30:100]     # 10-sns (30,115) (100,130)
	elif pre == 1:
	    cropped=img2[250:300, 70:265]     # 20-sns (70,250) (265,300)
	elif pre == 2:    
	    cropped=img2[450:520, 130:430] #100-sns
	elif pre == 3:    
	    cropped = img2[150:200, 75:390] #500-sns    
	# cropped = img2[180:250, 80:550] #realmoney (75,150)(390 ,200)

	config = ("-l eng --oem 3 --psm 6")
	number = cv2.fastNlMeansDenoising(cropped, 10, 7,21)
	number[number > 170] = 255
	number[number <= 170] = 0

	# config = ("-l eng --oem 3 --psm 3")
	# kernel1 = np.ones((3, 3), np.uint8)
	# kernel2 = np.ones((5, 5), np.uint8)
	# number = cv2.morphologyEx(number, cv2.MORPH_CLOSE, kernel1)
	#number = cv2.morphologyEx(number, cv2.MORPH_OPEN, kernel2)
	# number = cv2.GaussianBlur(number, (3, 3), 0)       # For Smoothing

	text = pytesseract.image_to_string(number, config=config)
	print("text=",text)

	# text="7AH 433534"
	# --------------------counterfeit------------------
	# xtest = data

	xtest=cv2.imread(v, 0)
	xtest = cv2.resize(xtest,(256,256))
	xtest=xtest.reshape(-1,256,256,1)
	xtest = xtest.astype('float32')
	xtest = xtest/ 255.

	with graph.as_default():
	 decimg = autoencoder.predict(data, verbose=1)
	# decimg = autoencoder.predict(xtest)
	# xtest = data
	# decimg = autoencoder._make_predict_function(xtest)

	mse1 = np.mean(np.power(xtest - decimg, 2),axis=1)
	mse0 = np.mean(mse1, axis=1)
	print(mse0[0])
	res=mse0[0]
	con=0
	if res > 0.002 :
		if res < 0.01:
			con=1

	if con == 0:
		return "-0-----------";	
	# ----------------------------------
	#  (denomination - 0/1/2/3) + (counterfeit= 0/1) + (sno)
	ret = str(result[0][0]) + str("1") + text;
	print("ret= ",ret);
	return ret;
    
    

# @app.route('/success', methods = ['POST'])  
# def success():  
#     if request.method == 'POST':  
#         f = request.files['file']  
#         f.save(f.filename)  
#         return render_template("index.html", name = f.filename)  

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=4000)
