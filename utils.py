import openai
import pandas as pd
import numpy as np
import json

# TODO: Improve the questions and the prompt to help the model categorize the responses into each pillar.
# TODO: Add more questions to the survey (10 max for prototype).
# TODO: Add evaluation function to evaluate model performance. (runtime, hallucinations, etc.)

## -- Utility functions -- ##
def atlas_analysis_gpt4(api_key: str, 
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
            completion = openai.ChatCompletion.create(model="gpt-4",
                                                  messages=[{"role": "system", "content":sys_prompt}, 
                                                            {"role": "user", "content":prompt}],
                                                  n=n,
                                                  max_tokens=max_tokens,
                                                  temperature=temperature)
            
            insight = completion.choices[0].message.content
            
        except openai.error.OpenAIError as e:
            raise ValueError(f"An error occurred while calling the OpenAI API: {e}")
                
        return insight
    
def atlas_analysis_davinci(api_key: str, 
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
            completion = openai.Completion.create(engine="text-davinci-003",
                                                prompt= sys_prompt + prompt,
                                                temperature=temperature,
                                                n=n,
                                                max_tokens=max_tokens)
            
            
            insight = completion.choices[0].text
            
        except openai.error.OpenAIError as e:
            raise ValueError(f"An error occurred while calling the OpenAI API: {e}")
                
        return insight
    
## -- End of utility functions -- ##