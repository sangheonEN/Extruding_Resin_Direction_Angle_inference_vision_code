# Semantic-Segmentation-with-Convolutional for predict line of extruded resin

<hr/>

Segmentation using sequence image data.

![image](https://user-images.githubusercontent.com/69128174/189470084-a111684a-37d8-4f12-abbb-6599ecba5719.png)

![image](https://user-images.githubusercontent.com/69128174/189470100-7be49874-e665-49cd-9af7-1b34bb9dadda.png)


외곽 선분 영역 점좌표 2D(W, H) 데이터를 주성분분석을 통해 고유벡터를 추출하여 2D상 외곽영역 점들의 고유한 방향을 추론
![image](https://user-images.githubusercontent.com/69128174/189470104-8c4b17e0-4426-4f4f-abd0-d4806b009459.png)

<hr/>

backbone: resnet101, 50
model architecture ASPP(Atrous Spatial Pyramid Pooling)

![image](https://user-images.githubusercontent.com/69128174/189470142-c75441a2-5b8d-4945-a2c2-ac2983d56df6.png)


coco dataset Image Pre-Processing

<hr/>

Folder image_task
       - binary image for threshold method, angle extraction method
       
Folder segmentation_model
       - fcn, deeplabv3 segmentation
       - loss function -> cross entropy, focal, effective number, my_focal
       
       my focal loss: softmax logits의 변량 편차를 높여 확실한 판단 확률 변량을 가지도록 지도하는 직관을 기반으로 합니다.
       
       ![image](https://user-images.githubusercontent.com/69128174/189470166-aa113081-92d7-4e21-be66-7e26a7715418.png)


Folder convlstm
       - classification using regression thresholding
       
Folder video_task
       - video to image (fps) 
<hr/>

