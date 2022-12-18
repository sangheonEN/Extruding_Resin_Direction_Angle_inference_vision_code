# Extruding Resin Direction Angle Inference Pipeline

<hr/>

## 1. Defined segment of resin for semantic segmentation.

![image](https://user-images.githubusercontent.com/69128174/189470084-a111684a-37d8-4f12-abbb-6599ecba5719.png)
<dir align="center">
       Sequence Resin Data
</div>

![image](https://user-images.githubusercontent.com/69128174/189470100-7be49874-e665-49cd-9af7-1b34bb9dadda.png)
<dir align="center">
       Segmentation Label
</div>

## 2. Calculation of angle about extruding resin direction
![image](https://user-images.githubusercontent.com/69128174/208290194-0a2a7ac8-cbcb-407b-baa3-976d30ca95f6.png)
![image](https://user-images.githubusercontent.com/69128174/208290205-e9880b98-93b5-47c5-8ebd-92f182235f35.png)
<dir align="center">
       Eigenvector extraction using 2D point coordinate data of the line segment area.
       The angle calculation uses the atan function.
</div>

## 3. Proposal Semantic Segmentation Model Archtecture

![image](https://user-images.githubusercontent.com/69128174/208290282-2654733f-debd-4605-8d12-1ea66cb25caa.png)
<dir align="center">
       Backbone: Deeplabv3 (resnet101, Atrous Spatial Pyramid Pooling)
       Loss Function: Boundary Focal Loss (proposal method)
       coco dataset Image Pre-Processing
</div>

## 4. Boundary Focal Loss (Novelty)

![image](https://user-images.githubusercontent.com/69128174/208290378-03fdb368-11db-4c4c-b89f-9160afb91e76.png)
![image](https://user-images.githubusercontent.com/69128174/208290440-ae5ecfa7-5bf7-4363-a587-71301647bd45.png)
<dir align="center">
       English ver: Designation of a boundary area focus loss function that reflects a higher loss value to a sample with high uncertainty with a low standard deviation by specifying the boundary area adjacent to an object as the standard deviation of softmax logits and applying a weight calculation function that is inversely proportional to the standard deviation.
       Korean ver: 객체와 객체 사이에 인접한 경계 영역을 softmax logits의 표준편차로 특정하고 표준편차에 반비례하는 가중치 산정함수를 적용하여 표준편차가 낮은 불확실성이 높은 표본에 더 높은 손실 값을 반영하는 경계 영역 초점 손실함수 고안.
</div>

## 5. mIoU Metric result compare with other loss function

![image](https://user-images.githubusercontent.com/69128174/208290568-87099ddb-7b00-4fa7-8786-4ea4640522cb.png)
<dir align="center">
       Cross Balancing Loss, Cross Entropy, Focal Loss, Boundary Focal Loss result box plot.
</div>

![image](https://user-images.githubusercontent.com/69128174/208290860-622e011e-2856-43d8-966d-97481a5bee83.png)
<dir align="center">
       외곽선분 영역 Left, Right Line 분할 결과 비교 그래프.
</div>

## 6. Inference Result Image
![image](https://user-images.githubusercontent.com/69128174/208290959-115a9675-e895-4669-aaf9-0a5f668efde3.png)
<dir align="center">
       입력데이터, Segmentation 추론 결과, 방향 각도 계산 결과
</div>
