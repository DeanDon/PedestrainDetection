import cv2
import numpy as np
import os
import person_detect_api
class seriers_image(object):
    def __init__(self,path):
        self.path=path
        self.list_image=os.listdir(path)
        self.list_image.sort()
        self.__index=0
    def read(self):
        image=cv2.imread(self.path+self.list_image[self.__index])
        self.__index+=1
        return True,image
def load_dataset(path):
    file_list=os.listdir(path)
    file_list.sort()
    return file_list
def gen_bboxs(img,bbox):
    bbox_width=bbox[2]-bbox[0]
    bbox_height=bbox[3]-bbox[1]
    #print bbox
    bbox[0]=max(1,bbox[0]-int(bbox_width/3)+10)
    bbox[1]=max(1,bbox[1]-int(bbox_height/3)+10)
    bbox[2] =min(bbox[2]+int(bbox_width/3),img.shape[0]-10)
    bbox[3] = min(bbox[3]+int(bbox_height/3),img.shape[1]-10)
    result=img[bbox[1]:bbox[3],bbox[0]:bbox[2]]
    return bbox,result
def cal_overlap(gtbbox,bbox0):
    gtbbox=np.array(gtbbox,dtype=np.float)
    bbox0 = np.array(bbox0, dtype=np.float)
    ixmin = np.maximum(gtbbox[0], bbox0[0])
    iymin = np.maximum(gtbbox[1], bbox0[1])
    ixmax = np.minimum(gtbbox[2], bbox0[2])
    iymax = np.minimum(gtbbox[3], bbox0[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((bbox0[2] - bbox0[0] + 1.) * (bbox0[3] - bbox0[1] + 1.) +
           (gtbbox[2] - gtbbox[0] + 1.) *
           (gtbbox[3] - gtbbox[1] + 1.) - inters)
    overlaps = inters / uni
    return overlaps
#print cal_overlap([30,30,50,50],[35,35,45,45])
def compare_with(gtbbox,bbox0,bbox1):
    overlap0=cal_overlap(gtbbox,bbox0)
    overlap1=cal_overlap(gtbbox,bbox1)
    if overlap0>overlap1:
        return 1
    else:
        return 0

def main(ori_bbox):
    #dataset_root_dir='/home/dean/person_detect/'
    #filename_list=load_dataset(dataset_root_dir)
    #cap = cv2.VideoCapture('768x576.avi')
    path='/home/dean/VOT/car1/'
    cap=seriers_image(path)
    groud_truth=open(path+'groundtruth.txt').readlines()
    _, img = cap.read()
    #print img
    bbox_pre = ori_bbox

    net=person_detect_api.load_model()
    temp_bbox=[]
    index=0
    while _:
        #img=cv2.imread(dataset_root_dir+file_name)
        fine_tune_bbox,pre_frame=gen_bboxs(img,bbox_pre)

        # cv2.imwrite('test1.jpg',pre_frame)
        # cv2.waitKey()
        # exit()
        #print fine_tune_bbox
        #print pre_frame.shape
        #print fine_tune_bbox
        bbox_curr_=person_detect_api.detect(pre_frame,net)
        #print bbox_curr_

        #print bbox_curr_
        if bbox_curr_==None:
            #bbox_pre=temp_bbox
            _, img = cap.read()
            print 'ERROR'
            continue
        line = groud_truth[index].split(',')
        coord_groud_truth = [int(float(line[0])),
                             int(float(line[1])),
                             int(float(line[4])),
                             int(float(line[5])), ]
        if len(bbox_curr_)==1:
            bbox_curr = [coord for coord in bbox_curr_[0][:4]]
        if len(bbox_curr_)>1:
            overlap=-1.0
            max_index=0
            for i in range(len(bbox_curr_)):
                bbox_curr1 = [coord for coord in bbox_curr_[i][:4]]
                # bbox_curr1 = [coord for coord in bbox_curr_[1][:4]]

                bbox_curr1[0] = fine_tune_bbox[0] + bbox_curr1[0]
                bbox_curr1[2] = fine_tune_bbox[0] + bbox_curr1[2]+20
                bbox_curr1[1] = fine_tune_bbox[1] + bbox_curr1[1]
                bbox_curr1[3] = fine_tune_bbox[1] + bbox_curr1[3]+20
                overlap_ = cal_overlap(coord_groud_truth, bbox_curr1)
                print 'gt',coord_groud_truth
                print 'bbox_curr',bbox_curr1
                print overlap
                if overlap_>overlap:
                    overlap=overlap_
                    max_index=i
            bbox_curr = [coord for coord in bbox_curr_[0][:4]]
            #print "overlap",overlap

        score=bbox_curr_[0][-1]
        bbox_curr[0]=fine_tune_bbox[0]+bbox_curr[0]
        bbox_curr[2] = fine_tune_bbox[0] + bbox_curr[2]
        bbox_curr[1] = fine_tune_bbox[1] + bbox_curr[1]
        bbox_curr[3] = fine_tune_bbox[1] + bbox_curr[3]
        bbox_pre=[int(coord) for coord in bbox_curr]


        cv2.rectangle(img,
                      (int(bbox_curr[0]), int(bbox_curr[1])),
                      (int(bbox_curr[2]), int(bbox_curr[3])),
                      (0, 0, 255),thickness=2)
        cv2.rectangle(img,
                      (coord_groud_truth[0],coord_groud_truth[1]),
                      (coord_groud_truth[2],coord_groud_truth[3]),(0,255,0),thickness=2)
        index+=1#'/home/dean/VOT/iceskater1/groundtruth.txt'
        #temp_bbox=bbox_curr
        cv2.imshow('test',img)
        cv2.imwrite(path+'tracker/'+str(index)+'.jpg',img)
        cv2.waitKey(1)
        _, img = cap.read()
#main([163,53,207,227])  ice1
#main([195,208,221,319])
#main([170,24,195,84]) #          blanket
#main([811,265,828,448])
#main([348,162,350,213])
#main([487,513,523,630])
main([246,162,357,279])