import pandas as pd
import streamlit as st
from utils import atlas_analysis_davinci, atlas_analysis_chatGPT

# TODO: Move prototype's frontend to Javascript (Flask + React?).
# TODO: Add more text mining and NLP features. (finish smart_surveys package and import it here)

# Define session state variables
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 550
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.5
if "model" not in st.session_state:
    st.session_state.model = ""
if "tokens_used" not in st.session_state:
    st.session_state.tokens_used = 0
if "cost" not in st.session_state:
    st.session_state.cost = 0
    
st.title("Smart Surveys Demo")
st.write("A description of the prototype and Atlas intelligence goes here.")

empty_survey = pd.DataFrame()

questions = ["Please imagine a ladder with steps numbered from zero at the bottom to ten at the top. The top of the ladder represents the best possible life for you, and the bottom of the ladder represents the worst possible life for you. On which step of the ladder would you say you personally feel you stand at this time?  (1 = Strongly Disagree, 10 = Strongly Agree)",
             "On which step do you think you will stand about five years from now?  (1 = Strongly Disagree, 10 = Strongly Agree)",
             "Please describe, in a few sentences, your physical health and workout routine.",
             "Briefly, what do you perceive a meaningful or purposeful life to be?",
             "Agree or Disagree: I am aware of what brings me purpose and I am motivated to pursue it. (1 = Strongly Disagree, 10 = Strongly Agree)",
             "In a few sentences, think of a dear friend or relative and write how you would improve their day."]

questions_df = pd.DataFrame(questions, columns=['question'])
#print(questions)

answers = [6, 
           8,
           "I feel in good physical health and try to maintain a 6-day workout routine. I mix cardio and strength training. I also try to eat a healthy diet and get plenty of sleep. Lately I have been failing on this by relapsing to nicotine, which has affected my sleep, due to a period of stress.",
           "For me, purpose is all about nourishing curiosity. My curiosity drives me towards new, ever more exciting, places and challenges. I perceive meaning or purpose to be exactly this quest for new adventures, both physical and intellectual.",
           7,
           "My parents, specially my mom, loves when I call and share what I am up to at school or what new crazy adventures im planning. So, I try to always make at least a litte bit of time every day to call them."]

answers_df = pd.DataFrame(answers, columns=['response'])

survey_results = pd.concat([empty_survey, questions_df, answers_df], axis=1)

st.sidebar.title("Smart Surveys Demo")

st.sidebar.subheader("Credentials")
open_api_key = st.sidebar.text_input("OpenAI API key")

st.sidebar.subheader("Questions")
# question 1
st.sidebar.subheader("Question 1")
st.sidebar.write(f"{questions[0]}")
response_1 = st.sidebar.text_input("Response 1")

st.sidebar.subheader("Question 2")
st.sidebar.write(f"{questions[1]}")
response_2 = st.sidebar.text_input("Response 2")

st.sidebar.subheader("Question 3")
st.sidebar.write(f"{questions[2]}")
response_3 = st.sidebar.text_input("Response 3")

st.sidebar.subheader("Question 4")
st.sidebar.write(f"{questions[3]}")
response_4 = st.sidebar.text_input("Response 4")

st.sidebar.subheader("Question 5")
st.sidebar.write(f"{questions[4]}")
response_5 = st.sidebar.text_input("Response 5")

st.sidebar.subheader("Question 6")
st.sidebar.write(f"{questions[5]}")
response_6 = st.sidebar.text_input("Response 6")

#st.sidebar.subheader("Model")
#model = st.sidebar.selectbox("Select model", 
#                             ["Davinci", "chatGPT"])


# Add a sidebar for controlling model parameters
with st.sidebar:
    st.write("## Model Parameters")
    st.session_state.max_tokens = st.slider("Max Tokens", min_value=32, max_value=1000, value=st.session_state.max_tokens)
    st.session_state.temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=st.session_state.temperature, step=0.05)
    
    st.write("## Model Selection")
    st.session_state.model = st.selectbox("Engine", 
                                           ["text-davinci-003", "gpt-3.5-turbo", "gpt-4"])

    st.write("### Model Prices")
    st.markdown("- (GPT-3) text-davinci-003: $0.02 per 1000 tokens")
    st.markdown("- (chatGPT-3.5) gpt-3.5-turbo: $0.002 per 1000 tokens")
    st.markdown("- (chatGPT-4) gpt-4: $0.03 per 1000 tokens")

    st.write("## Session Stats")

    st.write("## Tokens Used So Far...")
    st.write(st.session_state.tokens_used)
    
    st.write("## Cost So Far...")
    st.info(f"${st.session_state.cost:.5f}")

if response_1 and response_2 and response_3 and response_4 and response_5 and response_6:
    # generate insights
    if st.button("Generate Insights"):
        #st.write("Generating insights...")
        st.spinner("Generating insights...")
        
        atlas_test_df = pd.DataFrame(survey_results)
        #print(atlas_test_df)

        if st.session_state.model == "text-davinci-003":
            
            insights, tokens_used, query_cost = atlas_analysis_davinci(api_key=open_api_key,
                                              model=st.session_state.model,
                                              data=atlas_test_df,
                                              n=1,
                                              max_tokens=st.session_state.max_tokens,
                                              temperature=st.session_state.temperature)

            st.session_state.tokens_used += tokens_used
            st.session_state.cost += query_cost
            
            st.write("## Generated Insights")
            st.write(f"\n {insights} \n")
            
            st.info(f"Query Tokens: {tokens_used}. Total tokens used: {st.session_state.tokens_used}")
            st.info(f"Query Cost: {query_cost}. Total Session's Cost ${st.session_state.cost:.5f}")

            st.balloons()
            
        elif st.session_state.model == "gpt-3.5-turbo":
            
            insights, tokens_used, query_cost = atlas_analysis_chatGPT(api_key=open_api_key,
                                                            model=st.session_state.model,
                                                            data=atlas_test_df,
                                                            n=1,
                                                            max_tokens=st.session_state.max_tokens,
                                                            temperature=st.session_state.temperature)

            st.session_state.tokens_used += tokens_used
            st.session_state.cost += query_cost
            
            st.write("## Generated Insights")
            st.write(f"\n {insights} \n")
            
            st.info(f"Query Tokens: {tokens_used}. Total tokens used: {st.session_state.tokens_used}")
            st.info(f"Query Cost: {query_cost}. Total Session's Cost ${st.session_state.cost:.5f}")

            st.balloons()
            
        elif st.session_state.model == "gpt-4":
            
            insights, tokens_used, query_cost = atlas_analysis_chatGPT(api_key=open_api_key,
                                                            model=st.session_state.model,
                                                            data=atlas_test_df,
                                                            n=1,
                                                            max_tokens=st.session_state.max_tokens,
                                                            temperature=st.session_state.temperature)

            st.session_state.tokens_used += tokens_used
            st.session_state.cost += query_cost
            
            st.write("## Generated Insights")
            st.write(f"\n {insights} \n")
            
            st.info(f"Query Tokens: {tokens_used}. Total tokens used: {st.session_state.tokens_used}")
            st.info(f"Query Cost: {query_cost}. Total Session's Cost ${st.session_state.cost:.5f}")

            st.balloons()            

else:
    st.warning("Please enter your responses to the questions.")