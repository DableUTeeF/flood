from torchreid.utils import FeatureExtractor
import hnswlib
import cv2
from mmdet.apis import inference_detector, init_detector
from mmcv import Config
from PIL import Image
from torchvision import transforms
from boxutils import add_bbox


def detect(detector, image):
    result = inference_detector(detector, image)[0]
    result = [x[x[:, -1] > 0.4] for x in result]  # filter confidence > 0.3
    result = result[0]  # class 0 is person
    images = []
    for box in result:
        x1, y1, x2, y2, _ = box.astype('int')
        images.append(image[y1:y2, x1:x2])
    return images, result


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    cfg = Config.fromfile('/media/palm/BiggerData/mmdetection/configs/cascade_rcnn/cascade_mask_rcnn_r101_fpn_mstrain_3x_coco.py')
    detector = init_detector(cfg,
                             '/media/palm/BiggerData/mmdetection/cp/cascade_mask_rcnn_r101_fpn_20e_coco_bbox_mAP-0.434__segm_mAP-0.378_20200504_174836-005947da.pth',
                             device='cuda')
    reid_extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path='/media/palm/BiggerData/deep-person-reid/cp/osnet_ms_d_c.pth.tar',
        device='cuda'
    )
    reid_ann = hnswlib.Index(space='cosine', dim=512)
    reid_ann.init_index(max_elements=2000, ef_construction=200, M=16)
    reid_ann.set_ef(50)

    reid_fetures = {}

    vid = cv2.VideoCapture('/media/palm/BiggerData/person_reid/VIRAT_S_000006.mp4')
    i = -1
    while vid.isOpened():
        _, frame = vid.read()
        i += 1
        cropped_images, boxes = detect(detector, frame)
        for j, image in enumerate(cropped_images):
            image = Image.fromarray(image)
            image = transform(image).unsqueeze(0)
            reid_vector = reid_extractor(image)
            if i==0:
                reid_ann.add_items(reid_vector[0].cpu().numpy())
            else:
                labels, distances = reid_ann.knn_query(reid_vector[0].cpu().numpy(), k=1)
                if distances[0] > 0.2:
                    reid_ann.add_items(reid_vector[0].cpu().numpy())
                frame = add_bbox(frame, boxes[j][:-1].astype(int), f'{labels[0, 0]}: {distances[0, 0]:.3f}', (0, 255, 0))
        cv2.imshow('a', frame)
        cv2.waitKey(15)
