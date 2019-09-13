import cv2
import model_yuz
import time
import numpy as np  


IMG_SIZE = 100
face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')

camera=cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))
model_out=2
def testet():
    
    i=1
    
    k=1
    
    while True:
        _,img=camera.read()
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x,y,w,h) in faces:
            a = "/home/ayse-pc/Desktop/imageclass/test3/"+str(i)+".jpg"
            #y=y-40
            #x=x-15
            if w>h:
                h=w
            else:
                w=h
            ekyh=int(h*0.25)
            ekxw=int(w*0.25)
            img=cv2.rectangle(img, (x-ekxw, y-ekyh), (x+w, y+h), (0, 255, 0), 2)
            #img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            #print "yuz"

            roi_gray = img[y:y+h, x:x+w]

            

            #roi_gray=cv2.resize(roi_gray,(256,384))

            
            #print "kic"
            cv2.imwrite(a,roi_gray)

            test_data=model_yuz.process_test_data()
            test_data = np.load('test_data3.npy')
            for data in (test_data[:i]):
                img_num = data[1]
                img_data = data[0]
                data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
                model_out = model_yuz.model.predict([data])[0]
                    #print model_out
            if len(faces)>1:
                break


        if len(faces)>0:
            '''if model_out[0][0]>0.8:
                cv2.putText(img,"ERKEK",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,225,255),1)
            elif model_out[0][0]<0.2:
                cv2.putText(img,"KADIN",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)'''

            if np.argmax(model_out) == 1:
                cv2.putText(img,"ERKEK",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,225,255),1)
            elif np.argmax(model_out)==0:
                cv2.putText(img,"KADIN",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
       

        
        out.write(img)

        cv2.imshow('img',img)
        #cv2.imshow("gray",roi_gray)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        #time.sleep(0.2)

        

    cap.release()
    out.release()
    cv2.destroyAllWindows()
