import pandas as pd
import os
from icecream import ic

work_dir = ic(os.getcwd())
data_dir = "dataloader"

tweets_dataloader = pd.read_csv(os.path.join(work_dir,data_dir,"all_tweets.csv"))
