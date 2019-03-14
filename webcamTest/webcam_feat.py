import numpy as np
import cv2
from random import randint
from keras.models import load_model

cascPath = "haarcascade_frontalface_default.xml"
cap = cv2.VideoCapture(0)
haar_face_cascade = cv2.CascadeClassifier(cascPath)
while(True):
    
    ret, frame = cap.read()
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	
    faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
    frame1=np.copy(frame)
    i=1
    for (x, y, w, h) in faces:
        	
        red=randint(0,255)
        green=randint(0,255)
        blue=randint(0,255)		
        cv2.rectangle(frame, (x, y), (x+w, y+h), (red, green, blue), 2)
        label="Person:"+str(i)
        cv2.putText(frame,label,(x+10,y+h+20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (red, green, blue), 2)
        i=i+1
		
	
    label=str(len(faces))+" faces detected."	
    cv2.putText(frame,label,(10,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    label="Press c to continue."
    cv2.putText(frame,label,(10,40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255), 2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        img=frame1
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

if len(faces)==0:
    print('No faces detected')
    exit()
	

cv2.imshow('image',frame)

k = cv2.waitKey(0) & 0xFF
if k == 27:       
    exit() 
    cv2.destroyAllWindows()
elif k == ord('c'): 
    cv2.destroyAllWindows()		
   
	
	
def getFaceImages(img,faces):
  faces_img=[]
	
  for (x, y, w, h) in faces:
  
    minx=x-50 
    maxx=x+w+50
    miny=y-50
    maxy=y+h+50
    crop_img = img[ miny:maxy,minx:maxx]
    faces_img.append(crop_img)
  
  return faces_img
  
faces_img=getFaceImages(img,faces)
print("Extracting Faces....", end =" ")
	
for i in range(len(faces_img)):
  
  faces_img[i]=cv2.resize(faces_img[i], (128,128))

faces_img=np.asarray(faces_img)
faces_img=faces_img/255
print("Done")
feat_dict={}
print("\nIdentifying Gender....", end =" ")  
model = load_model('models/model_Male.h5')
preds=model.predict(faces_img)
print('Done')
gen_list=[]
for i in range(faces_img.shape[0]):
    print("\nPerson"+str(i+1)+" (Gender)")
    
    if preds[i][0] >preds[i][1]:
      gen_list.append('Female')
      print("Female", end =" ")
      print(str(preds[i][0]*100)+" %")
    else:
      gen_list.append('Male')
      print("Male", end =" ")
      print(str(preds[i][1]*100)+" %")
	  
	  
  
feat_dict['Gender']=gen_list

print("\nIdentifying Race....", end =" ")   
model=load_model('models/model_Indian.h5')
preds0=model.predict(faces_img)
  
model=load_model('models/model_White.h5')
preds1=model.predict(faces_img)
  
model=load_model('models/model_Asian.h5')
preds2=model.predict(faces_img)
      
 
model=load_model('models/model_Black.h5')
preds3=model.predict(faces_img)

print('Done') 
race_list=[]
for i in range(faces_img.shape[0]):
    r=[]
    print("\nPerson"+str(i+1)+" (Race)")
    
    if preds0[i][1]>preds0[i][0]:
      r.append('Indian')
      print("Indian", end =" ")
      print(str(preds0[i][1]*100)+" %")
    if preds1[i][1]>preds1[i][0]:
      r.append('White')
      print('White', end =" ")
      print(str(preds1[i][1]*100)+" %")
    if preds2[i][1]>preds2[i][0]:
      r.append('Asian')
      print('Asian', end =" ")
      print(str(preds2[i][1]*100)+" %")
    if preds3[i][1]>preds3[i][0]:
      r.append('Black')
      print('Black', end =" ")
      print(str(preds3[i][1]*100)+" %")
    
      
    race_list.append(r)
      
feat_dict['Race']=race_list

print("\nIdentifying Expression....", end =" ")    
model=load_model('models/model_Smiling.h5')
preds0=model.predict(faces_img)
  
model=load_model('models/model_Frowning.h5')
preds1=model.predict(faces_img)
  
model=load_model('models/model_Mouth_Slightly_Open.h5')
preds2=model.predict(faces_img)
print('Done')      
  
exp_list=[]
for i in range(faces_img.shape[0]):
    e=[]
    print("\nPerson"+str(i+1)+" (Expression)")
    if preds0[i][1]>preds0[i][0]:
      e.append('Smiling')
      print("Smiling", end =" ")
      print(str(preds0[i][1]*100)+" %")
    if preds1[i][1]>preds1[i][0]:
      e.append('Frowning')
      print("Frowning", end =" ")
      print(str(preds1[i][1]*100)+" %")
    if preds2[i][1]>preds2[i][0]:
      e.append('Mouth Slightly Open')
      print("Mouth Slightly Open", end =" ")
      print(str(preds2[i][1]*100)+" %")
    exp_list.append(e)
    
      
feat_dict['Expression']=exp_list

print("\nIdentifying Age Group....", end =" ")    
model=load_model('models/model_Baby.h5')
preds0=model.predict(faces_img)
  
model=load_model('models/model_Child.h5')
preds1=model.predict(faces_img)
  
model=load_model('models/model_Youth.h5')
preds2=model.predict(faces_img)
      
  
  
model=load_model('models/model_Middle_Aged.h5')
preds3=model.predict(faces_img)
  
model=load_model('models/model_Senior.h5')
preds4=model.predict(faces_img)

print('Done')   
age_list=[]

for i in range(faces_img.shape[0]):
    a=[]
    print("\nPerson"+str(i+1)+" (Age Group)")
    if preds0[i][1]>preds0[i][0]:
      a.append('Baby')
      print("Baby", end =" ")
      print(str(preds0[i][1]*100)+" %")
    if preds1[i][1]>preds1[i][0]:
      a.append('Child')
      print("Child", end =" ")
      print(str(preds1[i][1]*100)+" %")
    if preds2[i][1]>preds2[i][0]:
      a.append('Youth')
      print("Youth", end =" ")
      print(str(preds2[i][1]*100)+" %")
    if preds3[i][1]>preds3[i][0]:
      a.append('Middle Aged')
      print("Middle Aged", end =" ")
      print(str(preds3[i][1]*100)+" %")
    if preds4[i][1]>preds4[i][0]:
      a.append('Senior')
      print("Senior", end =" ")
      print(str(preds4[i][1]*100)+" %")
    age_list.append(a)
    
      
feat_dict['Age Group']=age_list

print("\nIdentifying Hair....", end =" ") 
model=load_model('models/model_Black_Hair.h5')
preds0=model.predict(faces_img)
  
model=load_model('models/model_Brown_Hair.h5')
preds1=model.predict(faces_img)
  
model=load_model('models/model_Blond_Hair.h5')
preds2=model.predict(faces_img)
      
  
  
model=load_model('models/model_Gray_Hair.h5')
preds3=model.predict(faces_img)
  
model=load_model('models/model_Bald.h5')
preds4=model.predict(faces_img)

print('Done')
  
hair_list=[]
for i in range(faces_img.shape[0]):
    h=[]
    print("\nPerson"+str(i+1)+" (Hair)")
    if preds0[i][1]>preds0[i][0]:
      h.append('Black Hair')
      print("Black Hair", end =" ")
      print(str(preds0[i][1]*100)+" %")
    if preds1[i][1]>preds1[i][0]:
      h.append('Brown Hair')
      print("Brown Hair", end =" ")
      print(str(preds1[i][1]*100)+" %")
    if preds2[i][1]>preds2[i][0]:
      h.append('Blond Hair')
      print("Blond Hair", end =" ")
      print(str(preds2[i][1]*100)+" %")
    if preds3[i][1]>preds3[i][0]:
      h.append('Gray Hair')
      print("Gray Hair", end =" ")
      print(str(preds3[i][1]*100)+" %")
    if preds4[i][1]>preds4[i][0]:
      h.append('Bald')
      print("Bald", end =" ")
      print(str(preds4[i][1]*100)+" %")	  
	  
    hair_list.append(h)
    
      
feat_dict['Hair']=hair_list

print("\nIdentifying Wearing....", end =" ")  
model=load_model('models/model_Eyeglasses.h5')
preds0=model.predict(faces_img)
  
model=load_model('models/model_Wearing_Hat.h5')
preds1=model.predict(faces_img)
  
model=load_model('models/model_Wearing_Lipstick.h5')
preds2=model.predict(faces_img)
      
  
  
model=load_model('models/model_Wearing_Necktie.h5')
preds3=model.predict(faces_img)
 
print('Done')  
wear_list=[]
for i in range(faces_img.shape[0]):
    w=[]
    print("\nPerson"+str(i+1)+" (Wearing)")
    if preds0[i][1]>preds0[i][0]:
      w.append('Eyeglasses')
      print("Eyeglasses", end =" ")
      print(str(preds0[i][1]*100)+" %")
    if preds1[i][1]>preds1[i][0]:
      w.append('Hat')
      print("Hat", end =" ")
      print(str(preds1[i][1]*100)+" %")
    if preds2[i][1]>preds2[i][0] and feat_dict['Gender'][i]=="Female":
      w.append('Lipstick')
      print("Lipstick", end =" ")
      print(str(preds2[i][1]*100)+" %")
    if preds3[i][1]>preds3[i][0]:
      w.append('Necktie')
      print("Necktie", end =" ")
      print(str(preds3[i][1]*100)+" %")
    
    wear_list.append(w)
    
      
feat_dict['Wearing']=wear_list

print("\nIdentifying other features....", end =" ")   
model=load_model('models/model_5_O_Clock_Shadow.h5')
preds0=model.predict(faces_img)
  
model=load_model('models/model_Bangs.h5')
preds1=model.predict(faces_img)
  
model=load_model('models/model_Heavy_Makeup.h5')
preds2=model.predict(faces_img)
      
model=load_model('models/model_Mustache.h5')
preds3=model.predict(faces_img)
  
model=load_model('models/model_No_Beard.h5')
preds4=model.predict(faces_img)
  
model=load_model('models/model_Pale_Skin.h5')
preds5=model.predict(faces_img)
  
model=load_model('models/model_Rosy_Cheeks.h5')
preds6=model.predict(faces_img)
  
model=load_model('models/model_Sideburns.h5')
preds7=model.predict(faces_img)

print('Done')   
other_list=[]
for i in range(faces_img.shape[0]):
    print("\nPerson"+str(i+1)+" (Other Features)")
    o=[]
    if preds0[i][1]>preds0[i][0] and feat_dict['Gender'][i]=="Male":
      o.append('5 O Clock Shadow')
      print("5 O Clock Shadow", end =" ")
      print(str(preds0[i][1]*100)+" %")
    if preds1[i][1]>preds1[i][0]:
      o.append('Bangs')
      print("Bangs", end =" ")
      print(str(preds1[i][1]*100)+" %")
    if preds2[i][1]>preds2[i][0] and feat_dict['Gender'][i]=="Female":
      o.append('Heavy_Makeup')
      print("Heavy_Makeup", end =" ")
      print(str(preds2[i][1]*100)+" %")
    if preds3[i][1]>preds3[i][0] and feat_dict['Gender'][i]=="Male":
      o.append('Mustache')
      print("Mustache", end =" ")
      print(str(preds3[i][1]*100)+" %")
    if preds4[i][1]>preds4[i][0] and feat_dict['Gender'][i]=="Male":
      o.append('No Beard')
      print("No Beard", end =" ")
      print(str(preds4[i][1]*100)+" %")
    if preds5[i][1]>preds5[i][0]:
      o.append('Pale Skin')
      print("Pale Skin", end =" ")
      print(str(preds5[i][1]*100)+" %")
    if preds6[i][1]>preds6[i][0]:
      o.append('Rosy Cheeks')
      print("Rosy Cheeks", end =" ")
      print(str(preds6[i][1]*100)+" %")
    if preds7[i][1]>preds7[i][0] and feat_dict['Gender'][i]=="Male":
      o.append('Sideburns')
      print("Sideburns", end =" ")
      print(str(preds7[i][1]*100)+" %")
         
    other_list.append(o)
    
      
feat_dict['Other']=other_list

print('\n')  
print(feat_dict)    
 


image=img
white = np.zeros([image.shape[0],image.shape[1],3],dtype=np.uint8)
white.fill(255)
image= np.hstack( (image,white) )
white = np.zeros([50,image.shape[1],3],dtype=np.uint8)
white.fill(255)
image= np.vstack( (image,white) )

i=0
for (x, y, w, h) in faces:
  x0=img.shape[1]+((i*img.shape[1])//3)
  y0=20
  
  red=randint(0, 255)
  green=randint(0, 255)
  blue=randint(0, 255)
  
  minx=x-50 
  maxx=x+w+50
  miny=y-50
  maxy=y+h+50
  cv2.rectangle(image, (minx, miny), (maxx, maxy), (red, green, blue), 2)
  label="Person "+str(i+1)
  
  
  cv2.putText(image,label,(minx+10,maxy+20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (red, green, blue), 2)
 
  cv2.putText(image,label,(x0,y0),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (red, green, blue), 2)
  y0=y0+30
  
  label="Gender: "+feat_dict['Gender'][i]
  cv2.putText(image,label,(x0,y0),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (red, green, blue), 1)
  y0=y0+30
  
  lab_list=feat_dict['Race'][i]
  label='Race :'
  
  if len(lab_list)==0:
    label=label+"None"
    cv2.putText(image,label,(x0,y0),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (red, green, blue), 1)
    y0=y0+20
    
  else:
    cv2.putText(image,label,(x0,y0),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (red, green, blue), 1)
    y0=y0+20
    for j in range(len(lab_list)):
      cv2.putText(image,lab_list[j],(x0,y0),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (red, green, blue), 1)
      y0=y0+20
      
  y0=y0+10
  
  lab_list=feat_dict['Age Group'][i]
  label='Age Group :'
  
  if len(lab_list)==0:
    label=label+"None"
    cv2.putText(image,label,(x0,y0),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (red, green, blue), 1)
    y0=y0+20
    
  else:
    cv2.putText(image,label,(x0,y0),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (red, green, blue), 1)
    y0=y0+20
    for j in range(len(lab_list)):
      cv2.putText(image,lab_list[j],(x0,y0),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (red, green, blue), 1)
      y0=y0+20
      
  y0=y0+10
  
  lab_list=feat_dict['Expression'][i]
  label='Expression :'
  
  if len(lab_list)==0:
    label=label+"None"
    cv2.putText(image,label,(x0,y0),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (red, green, blue), 1)
    y0=y0+20
    
  else:
    cv2.putText(image,label,(x0,y0),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (red, green, blue), 1)
    y0=y0+20
    for j in range(len(lab_list)):
      cv2.putText(image,lab_list[j],(x0,y0),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (red, green, blue), 1)
      y0=y0+20
      
  y0=y0+10
  
  lab_list=feat_dict['Hair'][i]
  label='Hair :'
  
  if len(lab_list)==0:
    label=label+"None"
    cv2.putText(image,label,(x0,y0),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (red, green, blue), 1)
    y0=y0+20
    
  else:
    cv2.putText(image,label,(x0,y0),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (red, green, blue), 1)
    y0=y0+20
    for j in range(len(lab_list)):
      cv2.putText(image,lab_list[j],(x0,y0),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (red, green, blue), 1)
      y0=y0+20
      
  y0=y0+10
  lab_list=feat_dict['Wearing'][i]
  label='Wearing :'
  
  
  if len(lab_list)==0:
    label=label+"None"
    cv2.putText(image,label,(x0,y0),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (red, green, blue), 1)
    y0=y0+20
    
  else:
    cv2.putText(image,label,(x0,y0),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (red, green, blue), 1)
    y0=y0+20
    for j in range(len(lab_list)):
      cv2.putText(image,lab_list[j],(x0,y0),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (red, green, blue), 1)
      y0=y0+20
  
  
  y0=y0+10
  lab_list=feat_dict['Other'][i]
  label='Other :'
  
  
  if len(lab_list)==0:
    label=label+"None"
    cv2.putText(image,label,(x0,y0),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (red, green, blue), 1)
    y0=y0+20
    
  else:
    cv2.putText(image,label,(x0,y0),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (red, green, blue), 1)
    y0=y0+20
    for j in range(len(lab_list)):
      cv2.putText(image,lab_list[j],(x0,y0),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (red, green, blue), 1)
      y0=y0+20
	  
  i=i+1
      
  
  




 
cv2.imshow('image',image)

k = cv2.waitKey(0) & 0xFF
if k == 27:        
    cv2.destroyAllWindows()  