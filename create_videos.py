import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt
import cv2
plt.rcParams.update({'font.size': 20})

"""this script creates videos from images 
to visualize voltage regulation effects 

by: jorge
"""
script_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = pathlib.Path(script_path).joinpath("Results3_manualTaps_Jul-08-2022")

# to get file names
init_dir = pathlib.Path(parent_dir).joinpath("T1_0_T2_0_T3_0","voltage")
file_names = [fn for fn in os.listdir(init_dir) if fn.endswith('.png')]

for file in file_names:
    name, _ = os.path.splitext(file)

    # video initialization
    video_name = os.path.join(parent_dir, name + '.avi')
    frame = cv2.imread(os.path.join(init_dir, file))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('M','J','P','G'), 4, (width,height))

    for folder in next(os.walk(parent_dir))[1]: 

        print(f"{folder}-{file}")
        # img
        img_file = pathlib.Path(parent_dir).joinpath(folder, "voltage", file)
        imageText = cv2.imread(str(img_file))
        #org: Where you want to put the text
        org = (50,350)
        # write the text on the input image
        cv2.putText(imageText, folder, org, fontFace = cv2.FONT_HERSHEY_COMPLEX, 
                fontScale = 1.5, color = (250,225,100))
        video.write(imageText)
        
    cv2.destroyAllWindows()
    video.release()
