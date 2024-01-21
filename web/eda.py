import json
import os
import torch
from modified_BotRGCN.model import BotRGCN  
import json
from proces_data import Twibot20
from icecream import ic

cat_properties_path = os.path.join("Data", 'cat_properties_tensor.pt')
category_properties = torch.load(cat_properties_path)
ic(category_properties.shape)

num_properties_path = os.path.join("Data", 'num_properties_tensor.pt')
num_prop = torch.load(num_properties_path)
ic(num_prop.shape)

# # Load model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# embedding_size,dropout,lr,weight_decay=128,0.3,1e-3,5e-3
# model=BotRGCN(num_prop_size=6,cat_prop_size=11,embedding_dimension=embedding_size).to(device)
# # model.load_state_dict(torch.load('botrgcn_model.pth', map_location=torch.device('cpu')))
# model.eval()

# json_data = []   
# with open(r'D:\Works\PoliticalCampaignsSpamBotDetection\app_with_modified_BotRGCN\nobot_sample.json','r') as f:
#     for data in f:
#         json_data.append(json.loads(data))

# if not os.path.exists('./save_data/'):
#     os.mkdir('./save_data/')

# with open('./save_data/dev.json', 'w') as f:
#     json.dump(json_data, f)

# dataset=Twibot20(device=device,process=True,save=True)
# des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx=dataset.dataloader()

# output = model(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
# output=output.max(1)[1].to('cpu').detach().numpy()

# print(output)

# # df_test = pd.read_json(f'{root}Twibot-20/test.json')
# # df_test.dropna(subset='tweet',inplace=True)

# # print(df_test.iloc[0]['label'])

# # bot_sample = df_test[df_test['label'] == 1].iloc[0].to_json()
# # with open('bot_sample.json', 'w') as file:
# #     file.write(bot_sample)

# # nobot_sample = df_test[df_test['label'] == 0].iloc[0].to_json()
# # with open('nobot_sample.json', 'w') as file:
# #     file.write(nobot_sample)

# # ic(df_test.isna().sum())
# # ic(df_test.domain[0])
# # ic(df_test[df_test['domain'].apply(lambda x: 'Politics' in x)].domain[:5])

# # tweets = list(df_test['tweet'])

# # all_tweets = []
# # for _,tweet_list in tqdm(enumerate(tweets)):
# #     all_tweets.extend(tweet_list)

# # all_tweets_df = pd.DataFrame({
# #     'tweets' : all_tweets
# # })

# # ic(all_tweets[0])

# # all_tweets_df.to_csv("all_tweets.csv",index=False)


# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # dataset=Twibot20(root="Data/", device=device,process=True,save=True)



# # data = dataset.df_data
# # data.dropna(subset='tweet')
# # tweets = list(data['tweet'])
# # for _,tweet_list in tqdm(enumerate(tweets)):
# #     for tweet in tweet_list:
# #         ic(tweet)
