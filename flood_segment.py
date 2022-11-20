import cv2
from detectron2.engine import default_argument_parser, default_setup
from detectron2.config import get_cfg
from sparseinst import add_sparse_inst_config
from detectron2.engine import DefaultPredictor
import torch
import numpy as np
import os


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_sparse_inst_config(cfg)
    cfg.merge_from_file('configs/sparse_inst_r50vd_dcn_giam_aug.yaml')
    cfg.merge_from_list(['DATASETS.TRAIN',
                         "('pig_train',)",
                         'DATASETS.TEST',
                         "('pig_val',)",
                         'MODEL.SPARSE_INST.CLS_THRESHOLD',
                         '0.5',
                         'MODEL.WEIGHTS',
                         '/media/palm/BiggerData/Flood/checkpoints/sparseinst_crop/model_0049999.pth'])
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


if __name__ == '__main__':
    args = default_argument_parser()
    args = args.parse_args()
    print("Command Line Args:", args)
    cfg = setup(args)
    predictor = DefaultPredictor(cfg)

    src = '/media/palm/Data/traffy_data/flood'
    dst = '/media/palm/Data/traffy_data/output'

    # image = cv2.imread('/media/palm/Data/traffy_data/flood/00000/df3284d88bc91cdc5f4965a0a4d8a39a7d1c2336.jpg')
    # with torch.no_grad():
    #     outputs = predictor(image)
    # mask = outputs['instances']._fields['pred_masks'][0].cpu().numpy().astype('uint8')
    # m = np.where(mask[..., None], (0, 0, 252), image).astype('uint8')
    # cropped = cv2.addWeighted(image, 0.5, m, 0.5, 0)
    # cv2.imshow('a', cropped)
    # cv2.waitKey()

    for folder in os.listdir(src):
        for file in os.listdir(os.path.join(src, folder)):
            image = cv2.imread(os.path.join(src, folder, file))
            with torch.no_grad():
                outputs = predictor(image)
            if len(outputs['instances']._fields['scores']) > 0:
                mask = outputs['instances']._fields['pred_masks'][0].cpu().numpy().astype('uint8')
                m = np.where(mask[..., None], (0, 0, 252), image).astype('uint8')
                image = cv2.addWeighted(image, 0.5, m, 0.5, 0)
            cv2.imwrite(os.path.join(dst, file), image)
