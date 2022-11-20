import pandas as pd
import shutil
import os
from mmseg.apis import inference_segmentor, init_segmentor


def get_traffic_and_flood(row):
    types = row['type'].split(',')
    if not row['saved'] or len(os.path.basename(row['photo'])) < 5:
        return
    out = None
    if 'น้ำท่วม' in types:
        out = '/media/palm/Data/traffy_data/road'
    elif 'ถนน' in types or 'จราจร' in types or 'การเดินทาง' in types:
        out = '/media/palm/Data/traffy_data/road'
    if out is not None:
        shutil.copy(
            os.path.join('/media/palm/Data/traffy_data/images', os.path.basename(row['photo'])),
            os.path.join(out, os.path.basename(row['photo'])),
        )


if __name__ == '__main__':
    a = '/media/palm/Data/traffy_data/flood'
    configs = '/media/palm/BiggerData/mmsegmentation/configs/'
    checkpoints = '/media/palm/BiggerData/mmsegmentation/cp'
    df = pd.read_csv('csvs/teamchadchart.csv')
    df.fillna('', inplace=True)
    saved = []
    for i in range(100):
        dff = pd.read_csv(f'csvs/saved_{i:04d}.csv')
        saved.extend(dff['saved'])
    df['saved'] = saved
    # df.apply(get_traffic_and_flood, axis=1)
    cfg, cp = ('segformer/segformer_mit-b5_512x512_160k_ade20k.py',  'segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth')
    model = init_segmentor(
        os.path.join(configs, cfg),
        os.path.join(checkpoints, cp),
        device='cuda'
    )
    for i, row in df.iterrows():
        types = row['type'].split(',')
        if not row['saved'] or len(os.path.basename(row['photo'])) < 5 or not os.path.exists(os.path.join('/media/palm/Data/traffy_data/images', os.path.basename(row['photo']))):
            continue
        results = inference_segmentor(model, os.path.join('/media/palm/Data/traffy_data/images', os.path.basename(row['photo'])))
        road_mask = results[0] == 6

        if road_mask.sum() < road_mask.shape[0] * road_mask.shape[1] * 0.2:
            continue

        out = None
        if 'น้ำท่วม' in types:
            out = '/media/palm/Data/traffy_data/flood'
        elif 'ถนน' in types or 'จราจร' in types or 'การเดินทาง' in types:
            out = '/media/palm/Data/traffy_data/road'
        else:
            out = '/media/palm/Data/traffy_data/road'
        if out is not None:
            num = f'{i // 5000:05d}'
            if not os.path.exists(os.path.join(out, num)):
                os.makedirs(os.path.join(out, num))
            shutil.copy(
                os.path.join('/media/palm/Data/traffy_data/images', os.path.basename(row['photo'])),
                os.path.join(out, num, os.path.basename(row['photo'])),
            )

