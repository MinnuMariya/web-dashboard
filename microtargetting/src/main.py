from dotenv import load_dotenv
from langchain.llms import OpenAI

from langchain_utils.main import run_pipeline
from dataloader.load_tweets import tweets_dataloader
from langchain.llms import GooglePalm

# Load env variables
load_dotenv()


# Initialize GooglePalm language model
# model = GooglePalm(temperature=0)
model = OpenAI(model_name="text-davinci-003", temperature=0.0)

print("                ")
tweet = 'Just enjoyed a lovely day at the park with friends. The weather was perfect, and we had a great time playing games and having a picnic. 🌞🌳 #OutdoorFun #Friends' 
# tweet = tweets_dataloader.tweets[1]
run_pipeline(tweet,model)
