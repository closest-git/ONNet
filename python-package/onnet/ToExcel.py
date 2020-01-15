import numpy as np
import pandas as pd
import json
import glob
import argparse

def OnVisdom_json(param,title):
    search_str = f"{param['data_root']}{param['select']}"
    files = glob.glob(search_str)
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
    parser = argparse.ArgumentParser(description='Load json of visdom curves. Save to EXCEL!')
    parser.add_argument("keyword", type=str, help="keyword")
    parser.add_argument("root",  type=str, help="root")

    args = parser.parse_args()

    if hasattr(args,'keyword') and hasattr(args,'root'):
        keyword = args.keyword  # "WNet_mnist"
        data_root = args.root   #"F:/arXiv/Diffractive Wavenet - an novel low parameter optical neural network/"
        param = {"data_root":data_root,
             "select":f"{keyword}*.json"}
        OnVisdom_json(param,keyword)