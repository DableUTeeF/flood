from mmseg.apis import inference_segmentor, show_result_pyplot, init_segmentor
import os
import cv2
import numpy as np
import math


ls = ('cascade convnext', 'mask2former_panoptic', 'mask2former_isntance')

if __name__ == '__main__':
    configs = '/media/palm/BiggerData/mmsegmentation/configs/'
    checkpoints = '/media/palm/BiggerData/mmsegmentation/cp'
    cf = 'https://github.com/open-mmlab/mmdetection/blob/master/configs/mask2former/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic.py'
    cfg = '/media/palm/BiggerData/mmdetection/configs/convnext/cascade_mask_rcnn_convnext-s_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco.py'
    cp = 'cascade_mask_rcnn_convnext-s_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco_20220510_201004-3d24f5a4.pth'
    road = cv2.imread('/media/palm/Data/traffy_data/flood/00000/37b27353910e5d76058a54d32a547b80a7e63ee8.jpg')
    noroad = cv2.imread('/media/palm/Data/traffy_data/flood/00000/0a024513251dba9b10a149551bdaa2e8fb2a7cd5.jpg')
    model_list = [
        ('segformer/segformer_mit-b5_512x512_160k_ade20k.py',
         'segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth'),
        ('segmenter/segmenter_vit-b_mask_8x1_512x512_160k_ade20k.py',
         'segmenter_vit-b_mask_8x1_512x512_160k_ade20k_20220105_151706-bc533b08.pth'),
    ]
    image = road
    for cfg, cp in model_list:
        model = init_segmentor(
            os.path.join(configs, cfg),
            os.path.join(checkpoints, cp),
            device='cuda'
        )
        results = inference_segmentor(model, road)
        road_mask = results[0] == 6
        cv2.imshow('a', road_mask.astype('uint8') * 250)
        results = inference_segmentor(model, noroad)
        noroad_mask = results[0] == 6
        cv2.imshow('b', noroad_mask.astype('uint8') * 250)
        cv2.waitKey()
