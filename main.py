import pandas as pd
import numpy as np
import time
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import cv2
import keyboard
from PIL import Image
import os
import warnings
from IPython.display import clear_output
warnings.filterwarnings("ignore")

x = np.load('image.npz')['arr_0']
x = 255.0 - x
y = pd.read_csv('labels.csv')['labels']
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
nClasses = len(classes)

spc = 5
fig = plt.figure(figsize=(2*nClasses,(1+spc*2)))
idx_cls = 0

for cls in classes:
    idxs = np.flatnonzero(y==cls)
    idxs = np.random.choice(idxs, spc, replace=False)
    i=0
    for idx in idxs:
        p1 = plt.subplot(spc, nClasses, (i*nClasses+idx_cls+1))
        p1 = sns.heatmap(np.reshape(x[idx], (22, 30)), cmap=plt.cm.gray, xticklabels=False, yticklabels=False, cbar=False)
        p1 = plt.axis('off')
        i+=1
    idx_cls+=1

xtrain, xtest, ytrain, ytest = train_test_split(x,y,train_size=0.75, random_state = 1)

xtrainscale = xtrain/255
xtestscale = xtest/255

LR = LogisticRegression(solver='saga', multi_class='multinomial')
LR.fit(xtrainscale, ytrain)

ypred = LR.predict(xtestscale)
acc = accuracy_score(ytest, ypred)

print("Model Accuracy:", acc*100, '%')

cm = pd.crosstab(ytest, ypred, rownames=['Actual'], colnames=['Predicted'])
p2 = plt.figure(figsize=(10,10))
p2 = sns.heatmap(cm, annot=True, cbar=False, fmt='d')

cap = cv2.VideoCapture(0)

while(True):
  try:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    height, width = gray.shape
    upper_left = (int(width / 2 - 56), int(height / 2 - 56))
    bottom_right = (int(width / 2 + 56), int(height / 2 + 56))

    frame = cv2.rectangle(frame, upper_left, bottom_right, (0, 255, 0), 2)

    roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

    threshold = 150
    roi[roi>threshold] = 255
    roi[roi<=threshold] = 0

    im_pil = Image.fromarray(roi)
    image_bw = im_pil.convert('L')
    image_bw_resized_inverted = image_bw.resize((22, 30), Image.ANTIALIAS)

    test_sample = np.array(image_bw_resized_inverted).reshape(1,660)
    test_pred = LR.predict(test_sample)
    print("Predicted class is: ", test_pred)

    cv2.imshow('frame',frame)
    cv2.imshow('roi',roi)
    cv2.imshow('grayscale',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  except Exception as e:
    pass

cap.release()
cv2.destroyAllWindows()