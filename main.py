
from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2
import time
import shapely.geometry as sg
import shapely.ops as so 
import math
import os 

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box
from utils.data_aug import letterbox_resize
from shapely.geometry import Polygon

from model import yolov3

parser = argparse.ArgumentParser(description="YOLO-V3 video test procedure.")
parser.add_argument("input_video", type=str,
                    help="The path of the input video.")
parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True,
                    help="Whether to use the letterbox resize.")
parser.add_argument("--class_name_path", type=str, default="./data/coco.names",
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str, default="./data/darknet_weights/yolov3.ckpt",
                    help="The path of the weights to restore.")
parser.add_argument("--save_video", type=lambda x: (str(x).lower() == 'true'), default=False,
                    help="Whether to save the video detection results.")
args = parser.parse_args()

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)

color_table = get_color_table(args.num_class)

vid = cv2.VideoCapture(args.input_video)
video_frame_cnt = int(vid.get(7))
video_width = int(vid.get(3))
video_height = int(vid.get(4))
video_fps = int(vid.get(5))

if args.save_video:
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    videoWriter = cv2.VideoWriter('video_result.mp4', fourcc, video_fps, (video_width, video_height))
#if os.path.exists("percentage.txt"):
#    os.remove("percentage.txt")
#if os.path.exists("info_black_width_100_v1.txt"):
#    os.remove("info_black_width_100_v1.txt")

with tf.Session() as sess:
    input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
    yolo_model = yolov3(args.num_class, args.anchors)
    with tf.variable_scope('yolov3'):
        l1, l3, l5, l7, l9, l11, f_m_1, f_m_2, f_m_3 = yolo_model.forward(input_data, False)
    pred_feature_maps = f_m_1, f_m_2, f_m_3
    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

    pred_scores = pred_confs * pred_probs

    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=200, score_thresh=0.3, nms_thresh=0.45)

    saver = tf.train.Saver()
    saver.restore(sess, args.restore_path)
    #fileper=open("percentage.txt","a")
    info_new=open("verify_file.txt","a")
    for i in range(video_frame_cnt):
        ret, img_ori = vid.read()
        height_ori, width_ori = img_ori.shape[:2]
        size=height_ori*width_ori
        if args.letterbox_resize:
            img, resize_ratio, dw, dh = letterbox_resize(img_ori, args.new_size[0], args.new_size[1])
        else:
            height_ori, width_ori = img_ori.shape[:2]
            img = cv2.resize(img_ori, tuple(args.new_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.

        start_time = time.time()
        filen1=open('res_n1/n1_{}.txt'.format(i+1),'a')
        filen3=open('res_n3/n3_{}.txt'.format(i+1),'a')
        filen5=open('res_n5/n5_{}.txt'.format(i+1),'a')
        filer1=open('res_r1/r1_{}.txt'.format(i+1),'a')
        filer2=open('res_r2/r2_{}.txt'.format(i+1),'a')
        filer3=open('res_r3/r3_{}.txt'.format(i+1),'a')
        filef1=open('res_f1/f1_{}.txt'.format(i+1),'a')
        filef2=open('res_f2/f2_{}.txt'.format(i+1),'a')
        filef3=open('res_f3/f3_{}.txt'.format(i+1),'a')
        print("********",i,"-th frame")
        n1, n3, n5, r1, r2, r3, f1, f2, f3 = sess.run([l1, l3, l5, l7, l9, l11, f_m_1, f_m_2, f_m_3],feed_dict={input_data: img})
        f_total = f1, f2, f3
        data1=n1[0]
        filen1.write('# Array shape: {0}'.format(data1.shape))
        for data_slice in data1:
            np.savetxt(filen1,data_slice,fmt='%.3f')
            filen1.write('# New slice')
        data3=n3[0]
        filen3.write('# Array shape: {0}'.format(data3.shape))
        for data_slice in data3:
            np.savetxt(filen3,data_slice,fmt='%.3f')
            filen3.write('# New slice')
        data5=n5[0]
        filen5.write('# Array shape: {0}'.format(data5.shape))
        for data_slice in data5:
            np.savetxt(filen5,data_slice,fmt='%.3f')
            filen5.write('# New slice')
        data7=r1[0]
        filer1.write('# Array shape: {0}'.format(data7.shape))
        for data_slice in data7:
            np.savetxt(filer1,data_slice,fmt='%.3f')
            filer1.write('# New slice')
        data9=r2[0]
        filer2.write('# Array shape: {0}'.format(data9.shape))
        for data_slice in data9:
            np.savetxt(filer2,data_slice,fmt='%.3f')
            filer2.write('# New slice')
        data11=r3[0]
        filer3.write('# Array shape: {0}'.format(data11.shape))
        for data_slice in data11:
            np.savetxt(filer3,data_slice,fmt='%.3f')
            filer3.write('# New slice')
        data_f1=f1[0]
        filef1.write('# Array shape: {0}'.format(data_f1.shape))
        for data_slice in data_f1:
            np.savetxt(filef1,data_slice,fmt='%.3f')
            filef1.write('# New slice')
        data_f2=f2[0]
        filef2.write('# Array shape: {0}'.format(data_f2.shape))
        for data_slice in data_f2:
            np.savetxt(filef2,data_slice,fmt='%.3f')
            filef2.write('# New slice')
        data_f3=f3[0]
        filef3.write('# Array shape: {0}'.format(data_f3.shape))
        for data_slice in data_f3:
            np.savetxt(filef3,data_slice,fmt='%.3f')
            filef3.write('# New slice')
        filen1.close()
        filen3.close()
        filen5.close()
        filer1.close()
        filer2.close()
        filer3.close()
        filef1.close()
        filef2.close()
        filef3.close()
        boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})
        #boxes_, scores_, labels_ = [], [] ,[] #sess.run([boxes, scores, labels], feed_dict={input_data: img})
        end_time = time.time()

        # rescale the coordinates to the original image
        if args.letterbox_resize:
            boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
            boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
        else:
            boxes_[:, [0, 2]] *= (width_ori/float(args.new_size[0]))
            boxes_[:, [1, 3]] *= (height_ori/float(args.new_size[1]))
        boxes_[boxes_< 0] = 0
        count=i+1
        #get information on boxes
        res=np.arange(len(labels_)*7).reshape(len(labels_), 7)
        res=res.astype(np.float32)
        res[:,0]=np.around(np.ones(len(labels_))*count,decimals=0)
        res[:,1]=np.around(labels_,decimals=0)
        res[:,2]=np.around(scores_,decimals=3)
        res[:,3:7]=np.around(boxes_,decimals=3)
        #print(res)
        np.savetxt(info_new,res,fmt='%.3f')
       

        #height_ori, width_ori = img_ori.shape[:2]

        #print("Loop Time:", (end_time_loop - start_time_loop) * 1000)
        #print("scores:")
        #print(scores_)
        """print(r1)"""
        """for i in range(len(boxes_)):
            x0, y0, x1, y1 = boxes_[i]
            plot_one_box(img_ori, [x0, y0, x1, y1], label=args.classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100), color=color_table[labels_[i]])
        cv2.putText(img_ori, '{:.2f}ms'.format((end_time - start_time) * 1000), (40, 40), 0,
                    fontScale=1, color=(0, 255, 0), thickness=2)
        cv2.imshow('image', img_ori)"""
        if args.save_video:
            videoWriter.write(img_ori)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #fileper.close()
    info_new.close()
    vid.release()
    if args.save_video:
        videoWriter.release()





