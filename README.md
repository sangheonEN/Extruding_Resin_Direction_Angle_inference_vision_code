# Semantic-Segmentation-with-Convolutional for predict line of extruded resin

<hr/>
Segmentation using sequence image data.
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

