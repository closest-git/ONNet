import numpy as np
import pandas as pd
import json
import glob

def OnVisdom_json(param,title):
    files = glob.glob(f"{param['data_root']}{param['select']}")
    datas = []
    cols = []
    for i, file in enumerate(files):
        with open(file, 'r') as f:
            meta = json.load(f)
            curve = meta['jsons']['loss']['content']['data'][0]
            legend = meta['jsons']['loss']['legend']
            cols.append(legend[0])
            item = curve['y']
            datas.append(item)
    df = pd.DataFrame(datas)
    df = df.transpose()
    for i,col in enumerate(cols):
        df = df.rename(columns={i: col})
    path = f"{param['data_root']}{title}.xlsx"
    df.to_excel(path )

    print(df.head())


if __name__ == '__main__':
    keyword = "WNet_mnist"
    param = {"data_root":"F:/arXiv/Diffractive Wavenet - an novel low parameter optical neural network/",
             "select":f"{keyword}*.json"}
    OnVisdom_json(param,keyword)