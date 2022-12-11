import pandas as pd
import os

def download(row):
    saved = False
    if 'corrupt' not in row['photo'] and os.path.basename(row['photo']) not in os.listdir(dst):
        if 'น้ำท่วม' in row['type']:
            os.system(f'wget {row["photo"]} -P images/flood -T 5')
            saved = True
        elif 'ถนน' in row['type']:
            os.system(f'wget {row["photo"]} -P images/road -T 5')
            saved = True
    return saved


if __name__ == '__main__':
    dst = '/media/palm/Data/traffy_data/images'
    df = pd.read_csv('/home/palm/PycharmProjects/traffy/csvs/teamchadchart.csv')
    df.fillna('', inplace=True)
    saves = []
    for i in range(100):
        d = df.iloc[i*1000:(i+1)*1000]
        s = d.apply(download, axis=1)
        d['saved'] = s
        d = d[['saved']]
        d.to_csv(f'csvs/saved_{i:04d}.csv')
