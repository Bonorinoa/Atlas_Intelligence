import pandas as pd
import streamlit as st
from utils import atlas_analysis_davinci, atlas_analysis_gpt4

# TODO: Move prototype's frontend to Javascript (Flask + React?).
# TODO: Add more text mining and NLP features. (finish smart_surveys package and import it here)

st.title("Smart Surveys Demo")

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

st.sidebar.subheader("Model")
model = st.sidebar.selectbox("Select model", 
                             ["GPT-3", "GPT-4"])



if response_1 and response_2 and response_3 and response_4 and response_5 and response_6:
    # generate insights
    if st.button("Generate Insights"):
        #st.write("Generating insights...")
        st.spinner("Generating insights...")
        
        atlas_test_df = pd.DataFrame(survey_results)
        #print(atlas_test_df)

        if model == "GPT-3":
            
            insights = atlas_analysis_davinci(api_key=open_api_key,
                                                                data=atlas_test_df,
                                                                n=1,
                                                                max_tokens=500,
                                                                temperature=0.75)

            st.write(insights)
            st.balloons()
            
        elif model == "GPT-4":
            
            insights = atlas_analysis_gpt4(api_key=open_api_key,
                                                            data=atlas_test_df,
                                                            n=1,
                                                            max_tokens=650,
                                                            temperature=0.75)

            st.write(insights)
            st.balloons()

else:
    st.warning("Please enter your responses to the questions.")