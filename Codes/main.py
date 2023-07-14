# import the necessary packages
from Image_Map import Image_Mapper
import argparse
import imutils
import cv2
import time
import numpy as np

# Usefull Links : https://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching/

def Cards_Handle(images, Function_type):
    showMatches=False

    [Card1, Card2, Card1_in_scene] = images
    # Card1_in_scene = cv2.imread('../DataSets/Card1_in_scene.jpg')          # trainImage
    # cv2.imshow("Card 1", Card1)
    # cv2.imshow("Card 2", Card2)
    # cv2.imshow("Card1_in_scene", Card1_in_scene)
    # cv2.waitKey(0)
    # Card2_in_scene=Card1_in_scene
    # cv2.imwrite("../DataSets/Card2_in_scene.png", Card2_in_scene)

    # stitch the images together to create a panorama
    mapper = Image_Mapper()
    if Function_type == "Draw_Border":
        retval = mapper.Draw_Border(images, showMatches=showMatches)
        if ([retval] != None):
            if (showMatches == True):
                [Card1_in_scene_Border, vis] = retval
            else:
                Card1_in_scene_Border = retval
        else:
            Card1_in_scene_Border = list(Card1_in_scene)
            Card1_in_scene_Border = np.array(Card1_in_scene_Border)
        return Card1_in_scene_Border
    elif Function_type == "Image_Map":
        retval = mapper.Image_Map(images, showMatches=showMatches)
        if ([retval] != None):
            if (showMatches == True):
                [Card2_in_scene, vis] = retval
            else:
                Card2_in_scene = retval
        else:
            Card2_in_scene = list(Card1_in_scene)
            Card2_in_scene = np.array(Card2_in_scene)
        return Card2_in_scene
    elif Function_type == "Draw_Cube":
        retval = mapper.Draw_Cube(images, showMatches=showMatches)
        if ([retval] != None):
            if (showMatches == True):
                [Card1_in_scene_cube, vis] = retval
            else:
                Card1_in_scene_cube = retval
        else:
            Card1_in_scene_cube = list(Card1_in_scene)
            Card1_in_scene_cube = np.array(Card1_in_scene_cube)
        return Card1_in_scene_cube


def main():
    cap = cv2.VideoCapture(0)
    cols = 640
    rows = 480
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = 10
    Card1_in_scene_Border_video = cv2.VideoWriter('./Card1_in_scene_Border_video.avi', fourcc, fps, (cols, rows))
    Card2_in_scene_video = cv2.VideoWriter('./Card2_in_scene_video.avi', fourcc, fps, (cols, rows))
    Card1_in_scene_cube_video = cv2.VideoWriter('./Card1_in_scene_cube_video.avi', fourcc, fps, (cols, rows))
    Card1 = cv2.imread('../DataSets/Card1.jpg')  # queryImage
    Card2 = cv2.imread('../DataSets/Card2.jpg')  # queryImage
    # print(Card1.shape)
    start_frame_cnt=0

    while (True):
        ret, frame = cap.read()
        # cv2.imshow('frame', frame)
        # print(frame.shape)
        # cv2.waitKey(0)

        if(start_frame_cnt>20):
            # print("salam")
            Card1_in_scene=frame
            images = [Card1, Card2, Card1_in_scene]

            # Function_type="Draw_Border"
            # Function_type="Image_Map"
            Function_type="Draw_Cube"

            write_frame=Cards_Handle(images, Function_type)

            try:
                cv2.imshow('write_frame', write_frame)

                # Card1_in_scene_Border_video.write(write_frame)
                # Card2_in_scene_video.write(write_frame)
                Card1_in_scene_cube_video.write(write_frame)
            except:
                print("None type image has been Received!!!")

        else:
            start_frame_cnt+=1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(.001)

    cap.release()
    Card1_in_scene_Border_video.release()
    Card2_in_scene_video.release()
    Card1_in_scene_cube_video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()