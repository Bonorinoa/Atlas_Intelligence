import openai
import pandas as pd
import numpy as np
import json

# TODO: Improve the questions and the prompt to help the model categorize the responses into each pillar.
# TODO: Add more questions to the survey (10 max for prototype).
# TODO: Add evaluation function to evaluate model performance. (runtime, hallucinations, etc.)

## -- Utility functions -- ##
def compute_cost(tokens, engine):
    
    model_prices = {"text-davinci-003": 0.02, 
                    "gpt-3.5-turbo": 0.002, 
                    "gpt-4": 0.03}
    model_price = model_prices[engine]
    
    cost = (tokens / 1000) * model_price

    return cost

def atlas_analysis_chatGPT(api_key: str, 
                           model: str,
                           data: pd.DataFrame, 
                           n: int, 
                           max_tokens: int, 
                           temperature: float) -> str:
    
        '''
        Prompt function for the atlas_analysis function with the GPT-4 model.
        It takes the features (questions and responses for now) and,
        based on the prompt, generates a well-being assessment of the surveyed object emulating a persona described by sys_prompt.
        params:
            api_key: OpenAI API key
            data: dataframe containing the data to generate the insights from
            n: number of insights to generate
            max_tokens: maximum number of tokens to generate
            temperature: temperature for the model 
        '''
        
        if not isinstance(api_key, str):
            raise TypeError(f"Expected a string but got {type(api_key)}")
        if not api_key:
            raise ValueError("Please provide an API key")
        
        openai.api_key = api_key
        
        #features = data.columns.tolist()
        questions = data.question.to_list()
        responses = data.response.to_list()
        keywords = "[well-being, happiness, sadness, anger, fear, disgust, surprise, trust, anticipation, joy, sadness, love, hate, contentment, satisfaction, gratitude, compassion, empathy, kindness, generosity, sympathy, guilt, shame, pride, embarrassment, regret, hope, optimism, pessimism, confidence, self-esteem, self-worth, self-respect, self-confidence, self-consciousness, self-awarenes]"
 
        # topics that could be used to evaluate a given survey according to the PERMA+4 measures
        pillars = "[positive emotions, engagement, relationships, meaning, accomplishment, physical health, mindset, work environment, economic security]"
        report_structure = "1. Positive Emotions \n 2. Engagement \n 3. Relationships \n 4. Meaning \n 5. Accomplishment \n 6. Physical Health \n 7. Mindset \n 8. Work Environment \n 9. Economic Security"
 
        prompt = f"Given the following questions [{str(questions)}] and responses [{str(responses)}]," \
            + f" provide a well being assessment of the surveyed object based on the 9 pillars of Perma+4 framework {pillars}." \
            + f" Be descriptive, insightful, and calm. The output must be a report that associates the responses to the questions with the 9 pillars of Perma+4 framework." \
            + f" Here is an example of the desired structure {report_structure}"
            
        sys_prompt = f"You are a well-being expert. You are tasked with providing a well-being assessment of the surveyed object." \
            + f" You can use the following keywords to help you: {keywords}. \n"
         
        # this will change to ChatCompletion endpoint when their apir errors are fixed   
        try:
            completion = openai.ChatCompletion.create(model=model,
                                                  messages=[{"role": "system", "content":sys_prompt}, 
                                                            {"role": "user", "content":prompt}],
                                                  n=n,
                                                  max_tokens=max_tokens,
                                                  temperature=temperature)
            
            insight = completion.choices[0].message.content
            tokens_used = completion.usage.total_tokens
            query_cost = compute_cost(tokens_used, model)
            
        except openai.error.OpenAIError as e:
            raise ValueError(f"An error occurred while calling the OpenAI API: {e}")
                
        return (insight, tokens_used, query_cost)
    
def atlas_analysis_davinci(api_key: str,
                           model: str, 
                           data: pd.DataFrame, 
                           n: int, 
                           max_tokens: int, 
                           temperature: float) -> str:
    
        '''
        Test prompt for the atlas_analysis function
        params:
            api_key: OpenAI API key
            data: dataframe containing the data to generate the insights from
            n: number of insights to generate
            max_tokens: maximum number of tokens to generate
            temperature: temperature for the model 
        '''
        
        if not isinstance(api_key, str):
            raise TypeError(f"Expected a string but got {type(api_key)}")
        if not api_key:
            raise ValueError("Please provide an API key")
        
        openai.api_key = api_key
        
        #features = data.columns.tolist()
        questions = data.question.to_list()
        responses = data.response.to_list()
        keywords = "[well-being, happiness, sadness, anger, fear, disgust, surprise, trust, anticipation, joy, sadness, love, hate, contentment, satisfaction, gratitude, compassion, empathy, kindness, generosity, sympathy, guilt, shame, pride, embarrassment, regret, hope, optimism, pessimism, confidence, self-esteem, self-worth, self-respect, self-confidence, self-consciousness, self-awarenes]"
 
        # topics that could be used to evaluate a given survey according to the PERMA+4 measures
        pillars = "[positive emotions, engagement, relationships, meaning, accomplishment, physical health, mindset, work environment, economic security]"
        report_structure = "1. Positive Emotions \n 2. Engagement \n 3. Relationships \n 4. Meaning \n 5. Accomplishment \n 6. Physical Health \n 7. Mindset \n 8. Work Environment \n 9. Economic Security"
 
        prompt = f"Given the following questions [{str(questions)}] and responses [{str(responses)}]," \
            + f" provide a well being assessment of the surveyed object based on the 9 pillars of Perma+4 framework {pillars}." \
            + f" Be descriptive, insightful, and calm. The output must be a report that associates the responses to the questions with the 9 pillars of Perma+4 framework." \
            + f" Here is an example of the desired structure {report_structure}"
            
        sys_prompt = f"You are a well-being expert. You are tasked with providing a well-being assessment of the surveyed object." \
            + f" You can use the following keywords to help you: {keywords}. \n"
         
        # this will change to ChatCompletion endpoint when their apir errors are fixed   
        try:
            completion = openai.Completion.create(engine=model,
                                                prompt= sys_prompt + prompt,
                                                temperature=temperature,
                                                n=n,
                                                max_tokens=max_tokens)
            
            
            insight = completion.choices[0].text
            tokens_used = completion.usage.total_tokens
            query_cost = compute_cost(tokens_used, model)
            
        except openai.error.OpenAIError as e:
            raise ValueError(f"An error occurred while calling the OpenAI API: {e}")
                
        return (insight, tokens_used, query_cost)
    
    
def generate_goals(api_key: str,
                   model: str,
                   insights: str):
    '''
    Function to generate suggested goals given insights generated
    api_key: OpenAI API key
    model: model to use for generation
    insights: insights generated from the atlas_analysis_x function
    '''
    
    if not isinstance(api_key, str):
        raise TypeError(f"Expected a string but got {type(api_key)}")
    
    if not api_key:
        raise ValueError("Please provide an API key")
    
    openai.api_key = api_key
    
    sys_prompt = "You are a well-being expert. Your task is to identify and recommend goals that will help the surveyed object improve his well-being given the survey insights report. \n"
    prompt = f"Given the following insights [{insights}], provide three suggested goals for the surveyed object that will maximize his net benefit for the effort required to improve along the dimensions that need the most improvement."
    
    if model == "text-davinci-003":
        try:
            completion = openai.Completion.create(engine=model,
                                            prompt=sys_prompt + prompt,
                                            temperature=0.7,
                                            max_tokens=500)
            
            goals = completion.choices[0].text
            tokens_used = completion.usage.total_tokens
            query_cost = compute_cost(tokens_used, model)
            
        except openai.error.OpenAIError as e:
            raise ValueError(f"An error occurred while calling the OpenAI API: {e}")
        
    else:
        completion = openai.ChatCompletion.create(model=model,
                                                  messages=[{"role": "system", "content":sys_prompt}, 
                                                            {"role": "user", "content":prompt}],
                                                  temperature=0.7,
                                                  max_tokens=500)
        
        goals = completion.choices[0].message.content
        tokens_used = completion.usage.total_tokens
        query_cost = compute_cost(tokens_used, model)
    
    return (goals, tokens_used, query_cost)
## -- End of utility functions -- ##