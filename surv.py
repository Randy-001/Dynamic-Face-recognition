from flask import Flask,request,render_template,url_for,redirect,session
import urllib.request
import base64
from PIL import Image
from io import BytesIO
import re, time, base64
from flask_pymongo import PyMongo
app=Flask(__name__)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib

from mtcnn.mtcnn import MTCNN



app.config['MONGO_URI'] = "mongodb+srv://user_1:aakash4224user1@cluster0.fjw1u.mongodb.net/db1?retryWrites=true&w=majority"
mongo = PyMongo(app)


local = ""
clf_loc = "print-attack_ycrcb_luv_extraTreesClassifier.pkl"
clf = joblib.load(clf_loc)
clf2_loc = "replay-attack_ycrcb_luv_extraTreesClassifier.pkl"
clf2 = joblib.load(clf2_loc)




def image_to_hist(img):
  img = plt.imread(img)
  plt.imsave("logo.png", img)
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
    img.save(str(t) + '.jpg', "JPEG")
    return str(t)
@app.route('/')
def home():
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
        return redirect(url_for('.profile', messages=x.inserted_id))
@app.route('/login',methods=['GET','POST'])
def login():
    if(request.method=="GET"):
        return render_template('login.html')
    else:
        name=request.form['name']
        password=request.form['password']
        myquery = {"name": name, "password": password}
        mydoc = mongo.db.users.find(myquery)
        for a in mydoc:
            id = a["_id"]
        global local
        local = str(id)

        return redirect(url_for('.profile', messages=id,name = name))
        #user = mongo.db.users.objects(name=name,password=password).first()
        #if user:


        #else:
        #return render_template('login.html',loginerror="Invalid username and password")



@app.route('/profile')
def profile():
    messages = request.args['messages']
    cmp =  request.args['name']

    return render_template('profile.html',mess=messages,name = cmp)
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
        add_dict={"name":emp_name,"phone":phone,"email":email,"job_position":job,"department":dept,"image":image.filename}
        for i in ld:

            mongo.db[str(i)].insert_one(add_dict)
        mongo.save_file(image.filename,image)
        return redirect(url_for('.profile', messages=id))
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
    return render_template("employee.html", users=users, companyid=id)


@app.route('/<id>/scan')
def scan(id):
    if local == id:
        return render_template('index.html',f=id)
    else:
        return "Please Login to access the current page"
@app.route('/process/<id>',methods=['GET','POST'])
def process(id):
    if request.method=="POST":
        img = request.form['f_img']
        images = []
        file_name=getI420FromBase64(img)
        file_name="./"+str(file_name)+".jpg"
        ld = []
        ld.append(id)
        print(file_name)
        #bool_op = liveness_predict(file_name, clf)
        #print(bool_op)
        result = DeepFace.verify("v_1.jpg", "v_2.jpg")
        print("Is verified: ", result["verified"])


        return "success"



if __name__ =="__main__":
    app.run(debug=True)