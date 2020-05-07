#%%
import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import os
import glob
from datetime import datetime
from helpers import get_domains, get_full_url, map_all_urls
from hilbertcurve.hilbertcurve import HilbertCurve

def plot_hilbert_curve(keyword):
    # get all pickle files for a keyword
    # keyword = "Angela_Merkel"
    files = glob.glob('analysis/data/**/kwMetadata_'+keyword+'.pkl', recursive=True)


    # %%
    # get previously computed urls
    with open("workingData/all_urls.pkl", "rb") as fp:
        all_urls = pickle.load(fp)

    #%% 

    #%% get hilbert curve and map urls to number
    # we have 2**(N*p) numbers on the curve
    N=2 # 2 dimensions
    # map urls to numbers
    url_map, p = map_all_urls(all_urls, N)
    hilbert_curve = HilbertCurve(p, N)

    # %%
    # iterate over all results
    x, y, c, pos = ([] for i in range(4))
    for day_index, file in enumerate(files):
        # import dataset
        df = pd.read_pickle(file)
        # remove null entries in results
        df["urls"] = df["urls"].apply(lambda x: list(filter(("null").__ne__, x)))
        # drop rows where lenght of results < 10
        df.drop(df[df["urls"].map(len) < 10].index, inplace=True)
        df.reset_index(inplace=True)
        # get full url
        # df["urls"] = df["result_hash"].apply(lambda x: get_full_url(hashmap[x]))
        #get hilbert curve for day
        curve = np.zeros(((len(df)*10,4)))
        for idx in range(len(df)):
            for jdx in range(10):
                x_coord,y_coord = hilbert_curve.coordinates_from_distance(url_map[df.urls[idx][jdx]])
                curve[idx*10+jdx, 0] = x_coord
                curve[idx*10+jdx, 1] = y_coord
                curve[idx*10+jdx, 2] = df.cluster[idx]
                curve[idx*10+jdx, 3] = 50/(jdx+1)
        x_day, y_day, c_day, pos_day = curve.T
        x.append(x_day)
        y.append(y_day)
        c.append(c_day)
        pos.append(pos_day)
        plt.scatter(x_day, y_day, c=c_day, s=pos_day, alpha=0.3)
        plt.ylim(0,1000)
        plt.xlim(0, 1000)
        date = datetime.utcfromtimestamp(df.timestamp[0]).strftime('%Y-%m-%d')
        plt.title(keyword + "    " + date)
        plt.savefig("analysis/plots/hilbert_"+keyword+"_"+str(day_index).zfill(2)+".png")
        plt.clf()

def merge_to_video(keyword):
    command = "ffmpeg -r 3 -f image2 -s 640x480 -i hilbert_"+keyword + "_%%02d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p " + keyword +".mp4"
    os.system(command)

if __name__ == "__main__":
    keywords = [
    'Katrin_Göring-Eckardt', 'Christian_Lindner', 'FDP', 'AfD',
    'Alexander_Gauland', 'Martin_Schulz', 'SPD', 'Sahra_Wagenknecht',
    'Alice_Weidel', 'CDU', 'Die_Linke', 'CSU', 'Angela_Merkel',
    'Dietmar_Bartsch', 'Cem_Özdemir', 'Bündnis90_Die_Grünen'
    ]
    for keyword in keywords:
        plot_hilbert_curve(keyword)


# convert to video
# http://hamelot.io/visualization/using-ffmpeg-to-convert-a-set-of-images-into-a-video/
# ffmpeg -r 2 -f image2 -s 640x480 -i hilbert_AfD_%02d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p test.mp4