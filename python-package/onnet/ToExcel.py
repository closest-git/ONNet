'''
@Author: Yingshi Chen

@Date: 2020-01-14 15:36:32
@
# Description: 
'''
import numpy as np
import pandas as pd
import json
import glob
import argparse
from scipy.signal import savgol_filter

def OnVisdom_json(param,title,smooth=False):
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
            if smooth:
                win = max(9,len(item)//10)
                cols.append(f"{legend[0]}_smooth")
                item_s = savgol_filter(item, win, 3)
                datas.append(item_s)
                pass
            
    df = pd.DataFrame(datas)
    df = df.transpose()
    for i,col in enumerate(cols):
        df = df.rename(columns={i: col})
        
    path = f"{param['data_root']}{title}_please_rename.xlsx"
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
        OnVisdom_json(param,keyword,smooth=True)
    else:
        param = {"data_root":"E:\Guided Inverse design of SPP structures\images",
             "select":f"3_4*.json"}
        OnVisdom_json(param,keyword)