from django.shortcuts import redirect, render
from django.views.generic import View
from django.contrib.auth import authenticate, login
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth.models import User, auth
from django.contrib import messages
from .models import bacsi as bs_model
from .models import benhnhan as benhnhan_model
import tensorflow as tf
from keras.models import load_model
import numpy as np
import cv2
# Create your views here.
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

def preprocess_image(image_path, desired_size=224):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_image_from_gray(img)
    img = cv2.resize(img, (desired_size, desired_size))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), desired_size / 30), -4, 128)

    return img
# from ThoBayMau.main import pred
model = tf.keras.models.load_model('models/model.h5')
img_path = r'ThoBayMau/media/images/test1.png'

def predict(request):
    fileObj = request.FILES.get('filePath', None)

    if fileObj is not None:
        fs = FileSystemStorage()
        filePathName = fs.save(fileObj.name, fileObj) # lưu file ảnh vào tệp media
        
        filePathName = fs.url(filePathName) # trỏ đường dẫn đến file cho phép hiển thị
        testimage = '.' + filePathName
        im_size = 320
        x_test = np.empty((1, im_size, im_size, 3), dtype=np.uint8)

        x_test[0, :, :, :] = preprocess_image(
            f'{img_path}',
            desired_size=im_size
        )
        preds = model.predict(x_test)
        context = {'filePathName' : filePathName, 'preds': preds}
        return render(request, 'predict.html', context)

    preds = model.predict(x_test)
    preds = np.argmax(round(preds))
    return render(request, 'predict.html', {'preds': preds})

def dashboard(request):
    return render(request, 'dashboard.html')

def Chatting(request):
    return render(request, 'chat.html')

def Profile(request):
    return render(request, 'profile.html')

def Calendar(request):
    return render(request, 'calendar.html')

def get_bacsi(request):
    bacsi_list = bs_model.objects.filter().order_by('bacsi_id')
    return render(request, 'doctors.html', {'bacsi_list': bacsi_list})
# def Doctors(request):
#     return render(request, 'doctors.html')

def AddDoctors(request):
    return render(request, 'add-doctor.html')

# def BenhNhan(request):
#     return render(request, 'BenhNhan.html')

def get_benhnhan(request):
    benhnhan_list = benhnhan_model.objects.filter().order_by('benhnhan_id')
    return render(request, 'BenhNhan.html', {'benhnhan_list': benhnhan_list})

def TinhHinhBenhNhan(request):
    return render(request, 'TinhTrangBenhNhan.html')

def EditProfile(request):
    return render(request, 'edit_profile.html')

def login_user(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = auth.authenticate(username=username, password=password)

        if user is not None:
            auth.login(request, user)
            return redirect('home')
        else:
            messages.info(request, 'Invalid Username or Password')
            return redirect('login_user')

    else:
        return render(request,'login.html')
    
def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        first_name = request.POST['first_name']
        last_name = request.POST['last_name']
        email = request.POST['email']
        password = request.POST['password']
        confirm_password = request.POST['confirm_password']

        if password==confirm_password:
            if User.objects.filter(username=username).exists():
                messages.info(request, 'Tên tài khoản đăng ký bị trùng xin vui lòng thử lại')
                return redirect(register)
            elif User.objects.filter(email=email).exists():
                messages.info(request, 'địa chỉ email này đã được đăng ký xin bạn thử lại')
                return redirect(register)
            else:
                user = User.objects.create_user(username=username, password=password, 
                                        email=email, first_name=first_name, last_name=last_name)
                user.save()
                
                return redirect('login_user')


        else:
            messages.info(request, 'Mật khẩu xác nhận không Trùng với mật khẩu ban đầu xin vui lòng thử lại!')
            return redirect(register)
            

    else:
        return render(request, 'register.html')

def ThongKe(request):
    all_results = Upload.objects.all()
    for i in all_results:
        print(i.imagespath)
        break
    # listOfImages = os.listdir('./media/')
    # listOfImagesPath = ['./media/' + i for i in listOfImages]
    context = { 'all_results':all_results}  #  'listOfImagesPath': listOfImagesPath,
    return render(request, 'analyst.html', context)

def Uploadfiles(request):
    return render(request, 'uploadfile.html')

import keras
import numpy as np
from PIL import Image
from django.core.files.storage import FileSystemStorage
from .models import Upload # .models sẽ trỏ đường dẫn đến models.py trong app folder ThoBayMau
from keras.preprocessing import image
import json
import tensorflow as tf
import os

'''
media = 'media'
model = keras.models.load_model('.h5')
'''

'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

img_height, img_width = 256, 256
with open('.models/labels.json', 'r') as f:
    labelInfo = f.read()
    
labelInfo = json.loads(labelInfo)
# print(Info)

tf.compat.v1.disable_eager_execution()
model_graph = tf.Graph()
with model_graph.as_default():
    tf_session = tf.compat.v1.Session()
    
    with tf_session.as_default():
        model = keras.models.load_model('.models/models.h5')
        '''
        
from .models import FileAnh as ImagesFile
def Dudoan(request):
    '''
    try:
        fileObj = request.FILES['filePath']
        fs = FileSystemStorage()
        
        filePathName = fs.save(fileObj.name, fileObj)
        filePathName = fs.url(filePathName)
        testimage = '.'+filePathName
         # print(testimage)
        # print(filePathName)

        # print(type(testimage))

        # if '%20' in testimage:
        #     testimage = fileObj.replace("%20", ' ')
        #     print(testimage)
        
        img = image.load_img(testimage, target=(img_height, img_width))
        test_image = img.img_to_array(img)
        test_image = np.expand_dims(test_image, axis=0)
        
        confidence = 0
        with model_graph.as_default():
            with tf_session.as_default():
                pred = model.predict(test_image)
                # print(pred)
                confidence = round(np.max(pred)*100, 2)
        predictedLabel = labelInfo[str(np.argmax(pred[0]))]
        print('Kết quả dự đoán: ', predictedLabel)
        print(f'Confidence: {confidence}%')
        
        filename = filePathName.split('/')[-1]
        print(filename)
        
        new_item = Upload(imagepath = filePathName, image = filename, predicted = predictedLabel, confidence = confidence)
        new_item.save()
        
        context = {'filePathName':filePathName, 'predictedLabel': predictedLabel, 'confidence': confidence, 'filename': filename}
        return render(request, 'predicts.html', context)
    except:
        return render(request, 'predicts.html')
    '''
    Images_List = ImagesFile.objects.filter().order_by('images_id')
    return render(request, 'predicts.html', {'Images_List' : Images_List})

''' Cách 2 https://www.youtube.com/watch?v=yGct2Wety_c
def Makepredictions(path):
    img = Image.open(path) # mở ảnh
    img_d = img.resize((244,244)) # chỉnh ảnh xuống cho model
    
    # kiểm tra hình ảnh có phải là RGB (ảnh màu) hay không
    if len(np.array(img_d).shape<4): 
        rgb_img = Image.new("RGB", img_d.size)
        rgb_img.paste(img_d)
    else:
        rgb_img=img_d
        
    # Chuyển ảnh vào mảng numpy và reshape
    rgb_img  = np.array(rgb_img, dtype=np.float64)
    rgb_img = rgb_img.reshape(1, 244, 244, 3)
    
    # Dự đoán
    
    predictions = model.predict(rgb_img)
    a = int(np.argmax(predictions))
    
    if a == 1:
        # thêm điều kiện sau trường hợp nào
        a = "Kết quả dự đoán: Có bệnh"
        
    
    elif a == 0:
        a = "Kết quả dự đoán: Không bệnh"

    return a
    
'''

'''
def Dudoan(request):
    if request.method == "POST" and request.FILES['upload']:
        if 'upload' not in request.FILES:
            err = "Không có hình ảnh nào được chọn"
            return render(request, 'predicts.html', {'err':err})
        f = request.FILES['upload']
        if f == '':
            err = "Không có hình ảnh nào được chọn"
            return render(request, 'predicts.html', {'err':err})
        upload = request.FILES['upload']
        fss = FILE
        
    print(request)
    print(request.POST.dict())
    return render(request, 'predicts.html')

'''

def index(request):
    return render(request, 'index.html')
