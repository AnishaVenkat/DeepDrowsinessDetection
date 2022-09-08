# DeepDrowsinessDetection
This project identifies if driver is awake or drowsy based on his/her facial expression.

1.) Using face landmarks library abd dlib we identify the points on the eyes , and the distance is calculated between the eye points if the distance threshold is less than 0.28 the points on the eye overlap which means the eyes are closed.

Libraries used :
numpy
opencv
dlib
scipy

Below gives us the accurate results:

![image](https://user-images.githubusercontent.com/26068822/189062585-e8bdc348-0ffb-4689-ad9f-3038445c7b2a.png)

![image](https://user-images.githubusercontent.com/26068822/189062947-f82f2346-72f9-41d3-acd0-89e7a968bd16.png)

2.) Using YOLOV5 Object detection:

    1.Using Web-cam I have collected images for awake and drowsy
    2.These images are passed yolov5s model for training purpose and accuracy is very good when compared to opencv dlib and face-segmentation library.
    
    



