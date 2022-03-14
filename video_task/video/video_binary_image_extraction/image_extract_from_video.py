"""

1. Video load

2. Each frame binary image extraction

3. Each frame binary image save

"""
import cv2
import os
import time

# read the video from path
video_name = "centering_coincidence.mp4"
video_path = os.path.abspath(os.path.dirname(__file__))
video_path = os.path.join(video_path, "video_data", video_name)
print(video_path)
save_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "binary_image")

cam = cv2.VideoCapture(video_path)

try:
    if not os.path.exists("binary_image"):
        os.makedirs("binary_image")

except Exception as e:
    print(e)

current_framge = 0
prev_time = 0
FPS = 24 # 영상 취득할때 사용한 카메라 프레임 FPS랑 맞춰야함.
count = 1

while(True):

    rval, frame = cam.read()
    key = cv2.waitKey()

    # FPS 계산하기 위해 초기 시간 저장.
    current_time = time.time() - prev_time

    if (rval): # 영상 취득할때 사용한 카메라 프레임 FPS랑 맞춰야함.

        prev_time = time.time()

        cv2.imwrite(save_path+"/"+video_name.split(".")[0]+"_"+"%04d"%count+".png", frame)
        count += 1

    if (cam.get(cv2.CAP_PROP_POS_FRAMES) == cam.get(cv2.CAP_PROP_FRAME_COUNT)):
        break


    if key == 27:
        break






# """
#
# 1. Video load
#
# 2. Each frame binary image extraction
#
# 3. Each frame binary image save
#
# """
# import cv2
# import os
# import time
#
# # read the video from path
# video_name = "centering_coincidence.mp4"
# video_path = os.path.abspath(os.path.dirname(__file__))
# video_path = os.path.join(video_path, "video_data", video_name)
# print(video_path)
# save_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "binary_image")
#
# cam = cv2.VideoCapture(video_path)
#
# try:
#     if not os.path.exists("binary_image"):
#         os.makedirs("binary_image")
#
# except Exception as e:
#     print(e)
#
# current_framge = 0
# prev_time = 0
# FPS = 24 # 영상 취득할때 사용한 카메라 프레임 FPS랑 맞춰야함.
# count = 1
#
# while(True):
#
#     rval, frame = cam.read()
#     key = cv2.waitKey()
#
#     # FPS 계산하기 위해 초기 시간 저장.
#     current_time = time.time() - prev_time
#
#     print(cam.get(cv2.CAP_PROP_POS_FRAMES))
#
#     if (rval) and (current_time > 1./FPS): # 영상 취득할때 사용한 카메라 프레임 FPS랑 맞춰야함.
#
#         prev_time = time.time()
#
#         cv2.imwrite(save_path+"/"+video_name.split(".")[0]+"_"+"%04d"%count+".png", frame)
#         count += 1
#
#     if (cam.get(cv2.CAP_PROP_POS_FRAMES) == cam.get(cv2.CAP_PROP_FRAME_COUNT)):
#         break
#
#
#     if key == 27:
#         break

