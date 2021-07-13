from flask import Flask,request,render_template,url_for,redirect,session
import urllib.request
import base64
from PIL import Image
import os
from io import BytesIO
import re, time, base64
from flask_pymongo import PyMongo
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
import numpy as np
import json
import smtplib
from flask_mail import Mail, Message
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import time
from datetime import date
today = date.today()
app=Flask(__name__)
mail = Mail(app)
app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = 'fernadezanthony13@gmail.com'
app.config['MAIL_PASSWORD'] = 'fernadezanthony@20'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)
model = load_model(r'C:\Users\USER\PycharmProjects\MiniHackathon\facenet_keras.h5')
c_name=""
email =""
file_name = ""
img_arr = []
info="allow"
#print(model(np.random.randn(1,160,160,3)))

app.config['MONGO_URI'] = "mongodb+srv://user_1:user_mongo@minihackathon.jyfgb.mongodb.net/db?retryWrites=true&w=majority"
mongo = PyMongo(app)
clf_loc = "print-attack_ycrcb_luv_extraTreesClassifier.pkl"
clf = joblib.load(clf_loc)
clf2_loc = "replay-attack_ycrcb_luv_extraTreesClassifier.pkl"
clf2 = joblib.load(clf2_loc)

def cosine_sim(a, b):
    d = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return 1 - d
def eucledian(a,b):
    return np.linalg.norm(a-b)


def get_face_vec(vec, model):
    #vec = preprocess_input(np.asarray(vec, 'float32'), version=2)
    return model(vec)

def get_embeddiing(path):
    img = plt.imread(path)
    detector = MTCNN()
    try:
        x, y, w, h = detector.detect_faces(img)[0]["box"]
    except Exception  as e:
        return render_template('index.html',f=id,name = name,info = "Kindly show your face properly")
    img = img[y:y + h, x:x + w]
    img = cv2.resize(img, (160, 160))
    img = np.expand_dims(img, axis=0)
    face_arr = get_face_vec(img, model)
    return face_arr[0]


def save_in_npy(ar, np_file):
    a = np.load(np_file)
    a = np.append(a, ar, axis=0)
    np.save(np_file, a)


def compare_embedding(a, b, tres=0.2):
    dist = cosine_sim(a, b)
    if dist <= tres:
        return True
    return False

def calc_hist(img):
    histogram = [0] * 3
    for j in range(3):
        histr = cv2.calcHist([img], [j], None, [256], [0, 256])
        histr *= 255.0 / histr.max()
        histogram[j] = histr
    return np.array(histogram)

def image_to_hist(img):
  img = plt.imread(img)
  detector = MTCNN()
  x,y,w,h = detector.detect_faces(img)[0]["box"]
  roi = img[y:y+h,x:x+w]
  img_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)
  img_luv = cv2.cvtColor(roi, cv2.COLOR_BGR2LUV)
  ycrcb_hist = calc_hist(img_ycrcb)
  luv_hist = calc_hist(img_luv)
  feature_vector = np.append(ycrcb_hist.ravel(), luv_hist.ravel())
  feature_vector = feature_vector.reshape(1, len(feature_vector))
  return feature_vector

def liveness_predict(img_path,clf):
  feature_vector = image_to_hist(img_path)
  prediction = clf.predict_proba(feature_vector)
  prediction_2 = clf2.predict_proba(feature_vector)
  print("----------->",prediction[0],"&&&&&&",prediction_2[0])
  op = prediction[0][1] + prediction_2[0][1]
  op = op/2
  if op >=0.7:
    return False
  return True

def getI420FromBase64(codec):
    base64_data = re.sub('^data:image/.+;base64,', '', codec)
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    t = time.time()
    print(img)
    img.save(os.path.join("./static/images/",str(t) + '.jpg'), "JPEG")
    return str(t)
@app.route('/')
def home():
    
    local =""
    cmp = ""
    img_arr.clear()
    return render_template('home.html')
@app.route('/details')
def details():
    return render_template('details.html')
@app.route('/signin',methods=["GET","POST"])
def signin():
    if(request.method=="GET"):
        return render_template("signin.html")
    else:
        company_name=request.form['name']
        password=request.form['password']
        email=request.form['email']
       
        dict={"name":company_name,"password":password,"email":email}
        x=mongo.db.users.insert_one(dict)
        dir = str(x.inserted_id)
        
        mongo.db.dates.insert_one({"c_id":dir})
        global local
        local = dir 
        current = r"C:\Users\USER\PycharmProjects\MiniHackathon\data"
        path = os.path.join(current,dir) 
        os.mkdir(path)
        
        return redirect(url_for('.profile', messages=x.inserted_id,name =company_name))
@app.route('/login',methods=['GET','POST'])
def login():
    if(request.method=="GET"):
        
        return render_template('login.html')
    else:
        name=request.form['name']
        password=request.form['password']
        myquery = {"name": name, "password": password}
        mydoc = mongo.db.users.find(myquery)
        global id
        for a in mydoc:
            id = a["_id"]
            if(id):
                 break
        else:
            return render_template('login.html',error = "Invalid username or password")
            
        global local
        local = str(id)
        current_date = today.strftime("%B %d, %Y")
        c = mongo.db.dates.find_one({'c_id': local})
        unit = 0
        current_date = current_date[:3]+current_date[-9:]
        for i in c:
            if i == current_date:
                break
        else:
            mongo.db.dates.update_one({'c_id': local}, { "$set": {current_date:[] }})

        
       
        
        global c_name
        c_name = name
        global img_arr
        img_arr=imgarr()
        return redirect(url_for('.profile', messages=id,name = name))
        #user = mongo.db.users.objects(name=name,password=password).first()
        #if user:


        #else:
        #return render_template('login.html',loginerror="Invalid username and password")

def imgarr():
    img_arr= []
    for i in os.listdir("./data/"+str(local)):
        img_arr.append({"name":str(i),"arr":np.array(get_embeddiing("./data/"+str(id)+"/"+str(i)))})
    return img_arr


@app.route('/profile')
def profile():
    
    messages = request.args['messages']
    cmp =  request.args['name']
    
    
    
    return render_template('profile.html',mess=messages,name = cmp)
@app.route('/spam')
def spam():

    print(file_name)

    return render_template('spam.html',file=file_name,id = local,name = c_name)

@app.route('/profile/add/<ObjectId:id>',methods=['GET','POST'])
def add(id):
    if(request.method=="GET"):
        return render_template('add.html',com=id)
    else:
        ld = []
        ld.append(id)
        emp_name=request.form['emp_name']
        phone=request.form['phone']
        email=request.form['email']
        dept = request.form['department']
        job=request.form['job']
        image = request.files['img']
        mongo.save_file(image.filename,image)
        #array  = np.array(get_embeddiing(image))
        
        add_dict={"name":emp_name,"phone":phone,"email":email,"job_position":job,"department":dept,"image":image.filename}
        
        for i in ld:

            y = mongo.db[str(i)].insert_one(add_dict)
        img = Image.open(image)
        img.save("./data/"+str(id)+"/"+str(y.inserted_id)+".jpg","JPEG")
        img_arr.append({"name":str(y.inserted_id),"arr":np.array(get_embeddiing("./data/"+str(id)+"/"+str(y.inserted_id)+".jpg"))})
        return redirect(url_for('.profile', messages=id,name=c_name))
@app.route('/profile/view/<ObjectId:id>')
def view(id):
    ld = []
    departments = []
    ld.append(id)
    for i in ld:
        arr = mongo.db[str(i)].find({})
    for i in arr:
        departments.append(i['department'])
    departments = list(set(departments))

    return render_template("details.html",dept = departments,companyid=id)
@app.route('/file/<filename>')
def file(filename):
    print("sx")
    return mongo.send_file(filename)

@app.route('/profile/view/<id>/<department>')
def employee(id,department):
    ld = []
    users = []
    ld.append(id)
    for i in ld:
        arr = mongo.db[str(i)].find({"department":department})
    for i in arr:
        users.append(i)
    users  = sorted(users, key = lambda i: i['name'])
    
    return render_template("view.html", users=users, companyid=id)


@app.route('/<id>/<name>/scan' ,methods = ["GET","POST"])
def scan(id,name):
    if request.method == "GET":
        if local == id:
            x = mongo.db.users.find_one({"name":name})
            print(x)
            global email
            email=x['email']
                
            return render_template('index.html',f=id,name = name)

            
        else:
            return "Please Login to access the current page"
    if request.method == "POST":
        global info
        info =  request.form['group1']
        
        
        return '', 204
@app.route('/<id>/sheet',methods=['GET','POST'])   
def sheet(id):
    if request.method == "POST":
        
        date = request.form['datePicker']
        x = mongo.db.dates.find({"c_id":id})
        attendance = []
        for i in x:
            print(i)
            try:
                attendance.append(i[str(date)])
            except:
                return render_template('table.html', ass = True)

        
        return render_template('table.html',attendance = attendance[0])



@app.route('/process/<id>/<name>',methods=['GET','POST'])
def process(id,name):
    if request.method=="POST":
        img = request.form['f_img']
        images = []
        global file_name
        file_name=getI420FromBase64(img)
        file_name="./static/images/"+str(file_name)+".jpg"
        
        
        
    
  
        
       
        try:
            bool_op = liveness_predict(file_name, clf)
        except Exception:
            return render_template('index.html',f=id,name = name,info = "Kindly show your face properly")

        if not bool_op:
            return render_template('index.html',f=id,name = name,info = "Kindly show your face properly")
                           
        m = get_embeddiing(file_name)
        m = np.array(m)
        cos = {}
        ans2 = []
        for n in img_arr:
            ans = eucledian(m,n["arr"])
            print(ans,n)
            ans2.append(ans)
            cos[str(ans)] = str(n["name"].replace(".jpg",""))
        mini = min(ans2)
        print(cos)
        ut = mongo.db[str(id)].find({})
        for i in ut:
            print("id",i['_id'])
            if(str(i['_id'])==cos[str(mini)]):
                user_name  = i["name"]
    
        '''
        print(cos)
        for n in cos:
            if n["ans"] == mini:
                print(n["name"])
            
                
        print(len(img_arr))'''
        print(email)

        if mini <= 9.0:
            
            named_tuple = time.localtime() 
            time_string = time.strftime("%H:%M:%S", named_tuple)
            new_tag = {
                "name":user_name,
                "time":time_string
            }
            current_date = today.strftime("%B %d, %Y")
            current_date = current_date[:3]+current_date[-9:]
            mongo.db.dates.update({'c_id': id}, {'$push': {str(current_date): new_tag}})
            return render_template('index.html',f=id,name = name,info = "You are permitted")
        connection = smtplib.SMTP('smtp.gmail.com',587)
        connection.ehlo()
        connection.starttls()
        tmessage = MIMEMultipart('alternative')
        tmessage['Subject'] = "Spam"
        tmessage['From'] = 'fernadezanthony13@gmail.com'
        tmessage['To'] = email
        html_msg = "<html><a href=http://127.0.0.1:5000/spam>Web page</a></html>"
        msg = MIMEText(html_msg,'html')
        tmessage.attach(msg)
                
        connection.login('fernadezanthony13@gmail.com','fernadezanthony@20')
        connection.sendmail("fernadezanthony13@gmail.com",email,tmessage.as_string())
        connection.quit()
        print("yes email send")
        while True:
                global info
                if(info=="permit" or info=="reject"):
                    break
        

        
        info = "You are "+ info +"ed" 


        return render_template('index.html',f=id,name = name,info=info)


if __name__ =="__main__":
    app.run(debug=True)
    