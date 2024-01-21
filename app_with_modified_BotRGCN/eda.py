from modified_BotRGCN.Dataset import Twibot20
import torch
from icecream import ic
import pandas as pd
from tqdm import tqdm

root="Data/"

df_test = pd.read_json(f'{root}Twibot-20/test.json')
df_test.dropna(subset='tweet',inplace=True)

ic(df_test.isna().sum())

tweets = list(df_test['tweet'])
all_tweets = []
for _,tweet_list in tqdm(enumerate(tweets)):
    all_tweets.extend(tweet_list)

all_tweets_df = pd.DataFrame({
    'tweets' : all_tweets
})

ic(all_tweets[0])

all_tweets_df.to_csv("all_tweets.csv",index=False)


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# dataset=Twibot20(root="Data/", device=device,process=True,save=True)



# data = dataset.df_data
# data.dropna(subset='tweet')
# tweets = list(data['tweet'])
# for _,tweet_list in tqdm(enumerate(tweets)):
#     for tweet in tweet_list:
#         ic(tweet)
