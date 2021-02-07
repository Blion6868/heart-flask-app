#!/usr/bin/env python
# coding: utf-8

# In[2]:


#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.layers import Dense, Dropout
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping


# In[4]:


df = pd.read_csv(r"C:\Users\bryan\Desktop\Heart_Disease_App\heart.csv")


# In[5]:


#dataset courtesy of Kaggle: https://www.kaggle.com/ronitf/heart-disease-uci


# In[6]:


df.head()


# In[7]:


len(df)


# In[8]:


X = df[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']].values
y = df['target'].values


# In[9]:


df.target.value_counts()


# In[10]:


sns.countplot(x="target", data=df)
plt.show()


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


# In[13]:


#scale data for better predictions

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[17]:


#define the Dense layers and activation functions

nn = keras.Sequential()

nn.add(Dense(30, activation='relu'))

nn.add(Dropout(0.2))

nn.add(Dense(15, activation='relu'))

nn.add(Dropout(0.2))

nn.add(Dense(1, activation='sigmoid'))

nn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss',mode='min', verbose=1, patience=25)

nn.fit(X_train, y_train, epochs = 1000, validation_data=(X_test, y_test),
         callbacks=[early_stop])


# In[18]:


#plot the model loss

model_loss = pd.DataFrame(nn.history.history)
model_loss.plot()


# In[19]:


nn.evaluate(X_test, y_test, verbose=0)


# In[20]:


nn.evaluate(X_train, y_train, verbose=0)


# In[21]:


pred = nn.predict(X_test)


# In[22]:


from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[23]:


mean_squared_error(y_test,pred)


# In[24]:


mean_absolute_error(y_test, pred)


# In[25]:


predictions = nn.predict_classes(X_test)


# In[26]:


#test the accuracy of the network

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))


# In[27]:


X_new = [[52,1,0,125,212,0,1,168,0,1.0,2,2,3]]
X_new = scaler.transform(X_new)
nn.predict_classes(X_new)


# In[28]:


X_new2 = [[59,1,1,140,221,0,1,164,1,0.0,2,0,2]]
X_new2 = scaler.transform(X_new2)
nn.predict_classes(X_new2)


# In[29]:


from tensorflow.keras.models import load_model
import joblib 


# In[30]:


nn.save("heart_disease_model_bry.h5")


# In[31]:


joblib.dump(scaler, 'heart_scaler_bry.pkl')


# In[32]:


heart_model = load_model("heart_disease_model_bry.h5")


# In[33]:


heart_scaler = joblib.load("heart_scaler_bry.pkl")


# In[34]:


heart_example = {"age":52,
                  "sex":1,
                 "cp":0,
                 "trestbps":125,
                 "chol":212,
                "fbs":0,
                "restecg":1,
                "thalach":168,
                "exang":0,
                "oldspeak":1.0,
                "slope":2,
                "ca":2,
                "thal":3}


# In[35]:


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
    oldpeak = sample_json['oldspeak']
    slope = sample_json['slope']
    ca = sample_json['ca']
    thal = sample_json['thal']

                        
    
    disease = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
    
    classes = np.array(['0', '1'])
    
    disease = scaler.transform(disease)
    
    class_ind = model.predict_classes(disease)
    
    return classes[class_ind]


# In[36]:


return_prediction(nn, heart_scaler, heart_example)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




