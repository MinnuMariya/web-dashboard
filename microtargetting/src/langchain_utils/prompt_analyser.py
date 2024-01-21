from langchain.prompts import PromptTemplate

def process_expert1_prompt(tweet:str, prompts:dict[str,str], model):

    opinion = ""
    for _, prompt_template in prompts.items():

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["tweet"],
        )

    # And a query intended to prompt a language model to populate the data structure.
    chain = prompt | model 
    output = chain.invoke({"tweet": tweet})

    opinion += str(output)

    return opinion

def process_expert2_prompt(tweet:str, prompt_template:str, model):

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["tweet"],
    )

    # And a query intended to prompt a language model to populate the data structure.
    chain = prompt | model 
    opinion = chain.invoke({"tweet": tweet})

    return opinion


def process_prediction_prompt(tweet, expert1_opinion, expert2_opinion, prompt_template, model):

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["tweet","expert1_opinion","expert2_opinion"],
    )

    # And a query intended to prompt a language model to populate the data structure.
    chain = prompt | model 

    prediction = chain.invoke({
        "tweet": tweet,
        "expert1_opinion": expert1_opinion,
        "expert2_opinion": expert2_opinion
    })

    return prediction