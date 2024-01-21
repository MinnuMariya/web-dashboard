from icecream import ic

from langchain_utils.prompts import expert1_prompts, expert2_prompts, prediction_prompt
from langchain_utils.prompt_analyser import (
    process_expert1_prompt, 
    process_expert2_prompt, 
    process_prediction_prompt
)

def run_pipeline(tweet, model):

    ic(tweet)

    expert1_opinion = process_expert1_prompt(tweet,expert1_prompts,model)
    expert2_opinion = process_expert2_prompt(tweet,expert2_prompts,model)
    prediction = process_prediction_prompt(
        tweet,
        expert1_opinion,
        expert2_opinion,
        prediction_prompt,
        model
    )

    ic(prediction)