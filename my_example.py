from flask import Flask, render_template, session, url_for, redirect
from tensorflow.keras.models import load_model
import numpy as np
from flask_wtf import FlaskForm
from wtforms import TextField, SubmitField
import joblib

def return_prediction(model, scaler, sample_json):
    
    age = sample_json["age"]
    sex = sample_json["sex"]
    cp = sample_json['cp']
    trestbps = sample_json['trestbps']
    chol = sample_json['chol']
    fbs = sample_json['fbs']
    restecg = sample_json['restecg']
    thalach = sample_json['thalach']
    exang = sample_json['exang']
    oldpeak = sample_json['oldpeak']
    slope = sample_json['slope']
    ca = sample_json['ca']
    thal = sample_json['thal']
    
    heart = [[age, sex, cp, trestbps, chol, fbs, restecg, 
    thalach, exang, oldpeak, slope, ca, thal]]
    
    classes = np.array(['0','1'])
    
    heart = scaler.transform(heart)
    
    class_ind = model.predict_classes(heart)
    
    return classes[class_ind][0]


app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'

class HeartForm(FlaskForm):
	
	age = TextField("age")
	sex = TextField("sex")
	cp = TextField('cp')
	trestbps = TextField('trestbps')
	chol = TextField('chol')
	fbs = TextField('fbs')
	restecg = TextField('restecg')
	thalach = TextField('thalach')
	exang = TextField('exang')
	oldpeak = TextField('oldpeak')
	slope = TextField('slope')
	ca = TextField('ca')
	thal = TextField('thal')
	submit = SubmitField('Predict')


@app.route("/",methods=['GET', 'POST'])
def index():
	
	form = HeartForm()

	if form.validate_on_submit():
		
		session['age'] = form.age.data
		session['sex'] = form.sex.data
		session['cp'] = form.cp.data
		session['trestbps'] = form.trestbps.data
		session['chol'] = form.chol.data
		session['fbs'] = form.fbs.data
		session['restecg'] = form.restecg.data
		session['thalach'] = form.thalach.data
		session['exang'] = form.exang.data
		session['oldpeak'] = form.oldpeak.data
		session['slope'] = form.slope.data
		session['ca'] = form.ca.data
		session['thal'] = form.thal.data


		return redirect(url_for("prediction"))
	return render_template('home.html',form=form)

heart_model = load_model("heart_disease_model_bry.h5")
heart_scaler = joblib.load("heart_scaler_bry.pkl")

@app.route('/prediction')
def prediction():
	
	content = {}

	content['age'] = float(session['age'])
	content['sex'] = float(session['sex'])
	content['cp'] = float(session['cp'])
	content['trestbps'] = float(session['trestbps'])
	content['chol'] = float(session['chol'])
	content['fbs'] = float(session['fbs'])
	content['restecg'] = float(session['restecg'])
	content['thalach'] = float(session['thalach'])
	content['exang'] = float(session['exang'])
	content['oldpeak'] = float(session['oldpeak'])
	content['slope'] = float(session['slope'])
	content['ca'] = float(session['ca'])
	content['thal'] = float(session['thal'])

	results = return_prediction(heart_model, heart_scaler, content)

	return render_template('prediction.html',results=results)

if __name__=='__main__':
	app.run()