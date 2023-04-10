import pandas as pd
import streamlit as st
from utils import atlas_analysis_davinci, atlas_analysis_chatGPT, generate_goals

# TODO: Move prototype's frontend to Javascript (Flask + React?).
# TODO: Add more text mining and NLP features. (finish smart_surveys package and import it here)

# Define session state variables
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 650
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.65
if "model" not in st.session_state:
    st.session_state.model = ""
if "tokens_used" not in st.session_state:
    st.session_state.tokens_used = 0
if "cost" not in st.session_state:
    st.session_state.cost = 0
    
st.title("Smart Surveys Demo")
st.write("A description of the prototype and Atlas intelligence goes here.")

empty_survey = pd.DataFrame()

questions = [
"I experience feelings of joy in a typical day (1-7)",
"Overall, I feel enthusiastic about my day to day life (1-7)",
"Please spend some time writing down the types of positive emotions you typically experience on a given day:",
"Describe the experiences that typically bring up these positive emotions for you:",
"Describe the experiences that typically block you from experiencing these emotions:",
"I typically feel fully absorbed in what I am doing (1-7)",
"Please spend some time describing examples of the tasks and experiences that you feel most absorbed with:",
"Please spend some time describing the extent to which you feel fully absorbed in a task or experience during a typical day:",
"Please spend some time describing what the experience of being fully absorbed in a task or experience feels like:",
"I am generally satisfied with the quality of relationships in my life (1-7)",
"Take a brief moment to think about your relationships with other people in your life, such as family, friends, significant others, colleagues at work or your broader social network.",
"Please spend spend some time writing  down how you feel about these relationships. You can choose to write about all the relationships or focus on specific areas of your social relationships:",
"In what ways do your social relationships add to the quality of your life:",
"In what ways do your social relationships detract from the quality of your life:",
"Take a brief moment to think about what is most important to you in life and why. ",
"What gives your life meaning?",
"What are you doing currently in life that is not meaningful?",
"Please spend some time writing about how meaningful you feel your life is:",
"Please spend some time writing about the purposes you feel your life serves:",
"My life is meaningful (1-7)",
"My life serves a purpose (1-7)",
"My life matters to the people I care about (1-7)",
"I typically accomplish what I set out to do (1-7)",
"I am generally satisfied with what I have accomplished in life (1-7)",
"Please spend some time writing  down things that you’ve accomplished. These do not need to be major accomplishments or anything that you’re especially proud of. They can be if you want, but if you can’t think of any that’s okay, just think of things that you’ve set out to do and successfully accomplished, no matter how big or how small they may seem to be:",
"Please spend at least two minutes writing down the how you feel about these accomplishments:",
"I typically feel physically healthy (1-7)",
"I feel in control of my physical health (1-7)",
"Please spend some time writing  down what physical health means to you:",
"Please spend some time describing how physically healthy you are",
"What do you do that contributes to your physical health",
"What do you do that impairs your physical health",
"What things that are out of your control do you feel impair your physical health",
"I believe I can improve my skills by working hard (1-7)",
"When I fail to improve at something, I try a new approach (1-7)",
"I can secure a worthwhile future by working hard (1-7)",
"Please spend at least two minutes writing down things you wish you were better at",
"When thinking of these things, do you typically judge your skill level against other people or against yourself in the past",
"I typically have access to natural light during the daytime (1-7)",
"I can conveniently access nature in my daily life (1-7)",
"I typically spend time in physical settings that are physically comfortable (1-7)",
"What types of physical environments do you most enjoy spending time in?",
"When are you able to spend time in these types of environments?",
"What stops you from spending more time in these types of environments?",
"I am comfortable with my current level of income (1-7)",
"I have enough savings to get through a financial emergency (1-7)",
"How would you describe your current financial status?",
"How do you feel about your current financial status?",
"Do you typically worry about money?"
]

questions_df = pd.DataFrame(questions, columns=['question'])
#print(questions)

answers = [
4, # I experience feelings of joy in a typical day (1-7)
5, # Overall, I feel enthusiastic about my day to day life (1-7)
"I typically experience happiness, contentment, and gratitude on a given day.", # Please spend some time writing down the types of positive emotions you typically experience on a given day:
"Spending time with loved ones, achieving a goal, and doing things I enjoy typically bring up positive emotions for me.", # Describe the experiences that typically bring up these positive emotions for you:
"Stressful situations, conflicts with others, and feeling overwhelmed typically block me from experiencing positive emotions.", # Describe the experiences that typically block you from experiencing these emotions:
6, # I typically feel fully absorbed in what I am doing (1-7)
"Writing, reading, and spending time outdoors are tasks and experiences that I feel most absorbed with.", # Please spend some time describing examples of the tasks and experiences that you feel most absorbed with:
"I feel fully absorbed in tasks or experiences for a few hours each day.", # Please spend some time describing the extent to which you feel fully absorbed in a task or experience during a typical day:
"Being fully absorbed in a task or experience feels like time passing quickly and feeling completely focused on the present moment.", # Please spend some time describing what the experience of being fully absorbed in a task or experience feels like:
6, # I am generally satisfied with the quality of relationships in my life (1-7)
"My relationships with family and friends are strong and positive, but I could improve my relationships with colleagues at work.", # Please spend spend some time writing down how you feel about these relationships. You can choose to write about all the relationships or focus on specific areas of your social relationships:
"My social relationships add to the quality of my life by providing support, love, and companionship.", # In what ways do your social relationships add to the quality of your life:
"My social relationships detract from the quality of my life when conflicts arise or when I feel overwhelmed by too many social obligations.", # In what ways do your social relationships detract from the quality of your life:
"What gives my life meaning is helping others, pursuing my passions, and making a positive impact on the world.", # What gives your life meaning?
"Currently, I feel like my job is not meaningful and I am searching for ways to find more purpose in my career.", # What are you doing currently in life that is not meaningful?
"I feel that my life is moderately meaningful, but there is room for improvement.", # Please spend some time writing about how meaningful you feel your life is:
"The purposes I feel my life serves are to help others, make a positive impact, and find joy in everyday experiences.", # Please spend some time writing about the purposes you feel your life serves:
5, # My life is meaningful (1-7)
6, # My life serves a purpose (1-7)
6, # My life matters to the people I care about (1-7)
6, # I typically accomplish what I set out to do (1-7)
4, # I am generally satisfied with what I have accomplished in life (1-7)
"Graduating college, learning to cook, and traveling to new places are things that I have accomplished.", # Please spend some time writing down things that you’ve accomplished. These do not need to be major accomplishments or anything that you’re especially proud of. They can be if you want, but if you can’t think of any that’s okay, just think of things that you’ve set out to do and successfully accomplished, no matter how big or how small they may seem to be:
"I feel proud of my accomplishments and they give me a sense of satisfaction.", # Please spend at least two minutes writing down the how you feel about these accomplishments:
7, # I typically feel physically healthy (1-7)
6, # I feel in control of my physical health (1-7)
"To me, physical health means feeling energized, strong, and free from illness or pain.", # Please spend some time writing down what physical health means to you:
"I am physically healthy and active, I exercise regularly, and eat a healthy diet.", # Please spend some time describing how physically healthy you are
"I exercise, eat healthy, and get enough sleep to contribute to my physical health.", # What do you do that contributes to your physical health
"Stress and occasional indulgences impair my physical health.", # What do you do that impairs your physical health
"Environmental factors such as pollution and allergens are out of my control and impair my physical health.", # What things that are out of your control do you feel impair your physical health
6, # I believe I can improve my skills by working hard (1-7)
7, # When I fail to improve at something, I try a new approach (1-7)
7, # I can secure a worthwhile future by working hard (1-7)
"I wish I were better at public speaking, cooking, and playing an instrument.", # Please spend at least two minutes writing down things you wish you were better at
"When thinking of these things, I typically judge my skill level against myself in the past and set goals to improve.", # When thinking of these things, do you typically judge your skill level against other people or against yourself in the past
7, # I typically have access to natural light during the daytime (1-7)
6, # I can conveniently access nature in my daily life (1-7)
7, # I typically spend time in physical settings that are physically comfortable (1-7)
"I most enjoy spending time in nature, cozy cafes, and quiet spaces.", # What types of physical environments do you most enjoy spending time in?
"I am able to spend time in these types of environments on weekends and during vacations.", # When are you able to spend time in these types of environments?
"Work obligations and city living often stop me from spending more time in these types of environments.", # What stops you from spending more time in these types of environments?
5, # I am comfortable with my current level of income (1-7)
4, # I have enough savings to get through a financial emergency (1-7)
"My current financial status is stable but I could improve my income and savings.", # How would you describe your current financial status?
"I feel content with my current financial status but also worry about saving for the future.", # How do you feel about your current financial status?
"I worry about money occasionally but try to focus on being financially responsible and planning for the future.", # Do you typically worry about money?
]

answers_df = pd.DataFrame(answers, columns=['response'])

survey_results = pd.concat([empty_survey, questions_df, answers_df], axis=1)

st.sidebar.title("Smart Surveys Demo")

st.sidebar.subheader("Credentials")
open_api_key = st.sidebar.text_input("OpenAI API key")


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

if len(survey_results) > 5:
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
            
            st.write("--------------------------------------------------------------")
            st.spinner("Generating goals...")
            
            goals, tokens_used, query_cost = generate_goals(api_key=open_api_key,
                                                            model=st.session_state.model,
                                                            insights=insights)
            
            st.session_state.tokens_used += tokens_used
            st.session_state.cost += query_cost
            
            st.write("## Generated Goals")
            st.write(f"\n {goals} \n")
            
            st.info(f"Query Tokens: {tokens_used}. Total tokens used: {st.session_state.tokens_used}")
            st.info(f"Query Cost: {query_cost}. Total Session's Cost ${st.session_state.cost:.5f}")
            
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
            
            st.write("--------------------------------------------------------------")
            st.spinner("Generating goals...")
            
            goals, tokens_used, query_cost = generate_goals(api_key=open_api_key,
                                                            model=st.session_state.model,
                                                            insights=insights)
            
            st.session_state.tokens_used += tokens_used
            st.session_state.cost += query_cost
            
            st.write("## Generated Goals")
            st.write(f"\n {goals} \n")
            
            st.info(f"Query Tokens: {tokens_used}. Total tokens used: {st.session_state.tokens_used}")
            st.info(f"Query Cost: {query_cost}. Total Session's Cost ${st.session_state.cost:.5f}")
            
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
            
            st.write("--------------------------------------------------------------")
            st.spinner("Generating goals...")
            
            goals, tokens_used, query_cost = generate_goals(api_key=open_api_key,
                                                            model=st.session_state.model,
                                                            insights=insights)
            
            st.session_state.tokens_used += tokens_used
            st.session_state.cost += query_cost
            
            st.write("## Generated Goals")
            st.write(f"\n {goals} \n")
            
            st.info(f"Query Tokens: {tokens_used}. Total tokens used: {st.session_state.tokens_used}")
            st.info(f"Query Cost: {query_cost}. Total Session's Cost ${st.session_state.cost:.5f}")

else:
    st.warning("Please enter your responses to the questions.")