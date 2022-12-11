import cv2
from detectron2.engine import default_argument_parser, default_setup
from detectron2.config import get_cfg
from sparseinst import add_sparse_inst_config
from detectron2.engine import DefaultPredictor
import torch
import numpy as np


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

    ims = ["/media/palm/BiggerData/Flood/bkk/310452208_420364323605056_5490563038462296835_n.jpg",
           "/media/palm/BiggerData/Flood/bkk/310775733_420364070271748_3763221270241511545_n.jpg",
           "/media/palm/BiggerData/Flood/bkk/310384936_420364446938377_9000783381814780377_n.jpg",
           "/home/palm/PycharmProjects/traffy/test.jpg"
           ]

    for im in ims:
        image = cv2.imread(im)
        with torch.no_grad():
            outputs = predictor(image)
        if len(outputs['instances']._fields['scores']) > 0:
            mask = outputs['instances']._fields['pred_masks'][0].cpu().numpy().astype('uint8')
            m = np.where(mask[..., None], (0, 0, 252), image).astype('uint8')
            image = cv2.addWeighted(image, 0.5, m, 0.5, 0)
        cv2.imshow('a', image)
        cv2.waitKey()

    # for folder in os.listdir(src):
    #     for file in os.listdir(os.path.join(src, folder)):
    #         image = cv2.imread(os.path.join(src, folder, file))
    #         with torch.no_grad():
    #             outputs = predictor(image)
    #         if len(outputs['instances']._fields['scores']) > 0:
    #             mask = outputs['instances']._fields['pred_masks'][0].cpu().numpy().astype('uint8')
    #             m = np.where(mask[..., None], (0, 0, 252), image).astype('uint8')
    #             image = cv2.addWeighted(image, 0.5, m, 0.5, 0)
    #         cv2.imwrite(os.path.join(dst, file), image)
