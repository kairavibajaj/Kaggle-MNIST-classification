#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


from copyreg import add_extension


df = pd.read_csv("/Users/kairavibajaj/Downloads/ift 6390/Kaggle competition 1/classification-of-mnist-digits/train.csv")
df.drop(['Unnamed: 1568'], axis = 1, inplace = True)
df


# In[57]:


half_df = df.iloc[: , 0:393] 

third_df = df.iloc[: , 785:1177]
first = pd.concat([half_df, third_df], axis =1 )
first


# In[58]:


plt.imshow(np.array(first.iloc[0, 1:]).reshape(28,28))


# In[23]:


plt.imshow(np.array(df.iloc[0, 1:]).reshape(28,56))


# In[5]:


df_y = pd.read_csv("/Users/kairavibajaj/Downloads/ift 6390/Kaggle competition 1/classification-of-mnist-digits/train_result.csv")
df_y.drop(['Index'], axis = 1 ,inplace=True)
df_y


# In[6]:


x_train = np.array(df.iloc[0:40000].T)
x_test = np.array(df.iloc[40000:].T)


# In[8]:


Y_train = np.array(df_y.iloc[0:40000].T)
Y_test = np.array(df_y.iloc[40000:].T)


# In[9]:


Y_train_0=(Y_train==0).astype(int)
Y_train_1=(Y_train==1).astype(int)
Y_train_2=(Y_train==2).astype(int)
Y_train_3=(Y_train==3).astype(int)
Y_train_4=(Y_train==4).astype(int)
Y_train_5=(Y_train==5).astype(int)
Y_train_6=(Y_train==6).astype(int)
Y_train_7=(Y_train==7).astype(int)
Y_train_8=(Y_train==8).astype(int)
Y_train_9=(Y_train==9).astype(int)
Y_train_10=(Y_train==10).astype(int)
Y_train_11=(Y_train==11).astype(int)
Y_train_12=(Y_train==12).astype(int)
Y_train_13=(Y_train==13).astype(int)
Y_train_14=(Y_train==14).astype(int)
Y_train_15=(Y_train==15).astype(int)
Y_train_16=(Y_train==16).astype(int)
Y_train_17=(Y_train==17).astype(int)
Y_train_18=(Y_train==18).astype(int)


# In[10]:


Y_test_0=(Y_test==0).astype(int)
Y_test_1=(Y_test==1).astype(int)
Y_test_2=(Y_test==2).astype(int)
Y_test_3=(Y_test==3).astype(int)
Y_test_4=(Y_test==4).astype(int)
Y_test_5=(Y_test==5).astype(int)
Y_test_6=(Y_test==6).astype(int)
Y_test_7=(Y_test==7).astype(int)
Y_test_8=(Y_test==8).astype(int)
Y_test_9=(Y_test==9).astype(int)
Y_test_10=(Y_test==10).astype(int)
Y_test_11=(Y_test==11).astype(int)
Y_test_12=(Y_test==12).astype(int)
Y_test_13=(Y_test==13).astype(int)
Y_test_14=(Y_test==14).astype(int)
Y_test_15=(Y_test==15).astype(int)
Y_test_16=(Y_test==16).astype(int)
Y_test_17=(Y_test==17).astype(int)
Y_test_18=(Y_test==18).astype(int)


# In[11]:


def sigmoid(z):
    s = 1/(1 + np.exp(-z))
    return s


# In[12]:


def initialize(dim):
    w = np.zeros(shape =(dim,1))
    b = 0
    return w,b


# In[13]:


def propagate(w,b,x,y):
    m = x.shape[1]
    A = sigmoid(np.dot(w.T, x) + b)
    cost = -1/m * np.sum(y * np.log(A) + (1-y)*np.log(1-A))
    dw = 1/m * np.dot(x , (A-y).T)
    db = 1/m * np.sum(A-y)
    grad = {"dw": dw, "db": db}
    return grad,cost


# In[14]:


def optimize(w,b,x,y, n_iter, learning_rate, print_cost = False):
    costs = []
    for i in range(n_iter):
        grad, cost = propagate(w,b,x,y)
        dw = grad["dw"]
        db = grad["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i%100 == 0:
            costs.append(cost)
        params = {"w":w, "b":b}
        grad = {"dw":dw , "db": db}
        return params, grad, costs


# In[15]:


def predict(w,b,x):
    m = x.shape[1]
    y_pred = np.zeros((1,m))
    w = w.reshape(x.shape[0], 1)
    A = sigmoid(np.dot(w.T, x) + b)
    for i in range(A.shape[1]):
        y_pred[:, i] = (A[:, i]>0.5) * 1
    return y_pred


# In[16]:


from sre_constants import NOT_LITERAL


def model(x_train, Y_train, x_test, Y_test, n_iter = 2000, learning_rate =0.5):
    w,b = initialize(x_train.shape[0])
    params, grads, costs = optimize(w, b, x_train, Y_train, n_iter, learning_rate)

    w = params["w"]
    b = params["b"]
    
    y_pred_train = predict(w,b, x_train)
    y_pred_test = predict(w,b,x_test)

    print("Train accuracy: {} %".format(100 - np.mean(np.abs(y_pred_train - Y_train)) * 100))
    print("Test accuracy: {} %".format(100 - np.mean(np.abs(y_pred_test - Y_test)) * 100))
    
    d = {"costs" : costs,
        "y_prediction_test" : y_pred_test,
        "y_prediction_train" : y_pred_train,
        "w" : w,
        "b" : b,
        "learning_rate" : learning_rate,
        "no_iterations" : n_iter}
    return d




# In[18]:


d = model(x_train, Y_train, x_test, Y_test, n_iter = 2000, learning_rate= 0.01)


# In[ ]:




