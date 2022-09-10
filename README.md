# Semantic-Segmentation-with-Convolutional for predict line of extruded resin

<hr/>
Segmentation using sequence image data.

![image](https://user-images.githubusercontent.com/69128174/189470084-a111684a-37d8-4f12-abbb-6599ecba5719.png)

![image](https://user-images.githubusercontent.com/69128174/189470100-7be49874-e665-49cd-9af7-1b34bb9dadda.png)

![image](https://user-images.githubusercontent.com/69128174/189470104-8c4b17e0-4426-4f4f-abd0-d4806b009459.png)

<hr/>

backbone: resnet101, 50
model architecture fcn, deeplabv3

coco dataset Image Pre-Processing

<hr/>

Folder image_task
       - binary image for threshold method, angle extraction method
       
Folder segmentation_model
       - fcn, deeplabv3 segmentation
       - loss function -> cross entropy, focal, effective number, my_focal

Folder convlstm
       - classification using regression thresholding
       
Folder video_task
       - video to image (fps) 
<hr/>

