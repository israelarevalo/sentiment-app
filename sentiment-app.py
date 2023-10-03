import streamlit as st 
import pandas as pd
# import ydata_profiling
# from streamlit_pandas_profiling import st_profile_report
from transformers import pipeline
import numpy as np
from app_store_scraper import AppStore

st.set_page_config(layout="wide")

with st.sidebar:
    st.image("https://cdn.pixabay.com/photo/2018/05/08/08/46/artificial-intelligence-3382509_1280.png", width=300)
    st.header("User Review Scraper and Sentiment Analysis Tool")
    st.info("This application is designed to facilitate scraping user reviews for mobile device applications and conduct sentiment analysis.")
    choice = st.radio("Navigation", ["Home", "Application Scraper", "Exploratory Data Analysis", "Sentiment Analysis"])

# Initialize scrape_df
if "scrape_df" not in st.session_state:
    st.session_state["scrape_df"] = None

# Initializing model for sentiment analysis
classifier = pipeline("text-classification", model = "j-hartmann/emotion-english-distilroberta-base", top_k = None)
emotions = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
predictions = []

# Home Page
if choice == "Home":
    st.header("Welcome to the User Review Scraper and Sentiment Analysis Tool")
    st.write("This application facilitates user review scraping and sentiment analysis.")
    st.write("Start by selecting 'Application Scraper,' filling out the required information, and scraping the desired user reviews. Then, click 'Run Exploratory Data Analysis' to generate a report of the data you uploaded (UNDER CONSTRUCTION). Lastly, click on the 'Sentiment Analysis' tab to run sentiment analysis on the reviews.")
    st.write("The citation for the sentiment analysis model used in this application is:")
    st.write("Jochen Hartmann, 'Emotion English DistilRoBERTa-base'. https://huggingface.co/j-hartmann/emotion-english-distilroberta-base/, 2022.")
    st.write("A paper for this model is currently in the works.")
    
    st.header("Current papers with base model:")
    st.write("Butt, S., Sharma, S., Sharma, R., Sidorov, G., & Gelbukh, A. (2022). What goes on inside rumour and non-rumour tweets and their reactions: A Psycholinguistic Analyses. Computers in Human Behavior, 107345.")  
    st.write("Kuang, Z., Zong, S., Zhang, J., Chen, J., & Liu, H. (2022). Music-to-Text Synaesthesia: Generating Descriptive Text from Music Recordings. arXiv preprint arXiv:2210.00434.")
    st.write("Rozado, D., Hughes, R., & Halberstadt, J. (2022). Longitudinal analysis of sentiment and emotion in news media headlines using automated labelling with Transformer language models. Plos one, 17(10), e0276367.")

# Application Scraper
if choice == "Application Scraper":
    st.header("Application Scraper")
    st.write("This tool allows you to scrape user reviews for mobile device applications from the Apple App Store. Simply enter the application ID and the number of reviews you would like to scrape, and click 'Scrape Reviews'.")
    app_id = st.text_input("Enter the application ID (ex. '1012822112')")
    num_reviews = st.text_input("Enter the number of reviews you would like to scrape (increments of 20)")
    specify_date_range = st.checkbox("Specify Date Range")

    if specify_date_range is True:
        from_date = st.date_input("Select the 'From' Date")
    else:
        from_date = None

    st.warning("Be sure to enter a valid application ID and verify its accuracy before clicking 'Scrape Reviews'.")
    if st.button("Scrape Reviews"):
        with st.spinner("Scraping Reviews..."):
            scrape = AppStore(country='us', app_name=None, app_id=app_id)
            st.session_state["app_id"] = app_id
            scrape.review(how_many=int(num_reviews))
            st.balloons()
            scrape_df = pd.DataFrame(np.array(scrape.reviews), columns=['reviews'])
            scrape_df = scrape_df.join(pd.DataFrame(scrape_df.pop('reviews').tolist()))
            if from_date is not None:
                scrape_df['date_formated'] = pd.to_datetime(scrape_df['date']).dt.date
                scrape_df = scrape_df[scrape_df['date_formated'] > from_date]
                scrape_df = scrape_df.drop(columns=['date_formated'])
                st.write("Here's a sample of your scraped reviews:")
                st.dataframe(scrape_df.head())
                st.session_state["scrape_df"] = scrape_df
                st.session_state["scrape_n"] = scrape_df.shape[0]
                st.write(f"You have successfully scraped {scrape_df.shape[0]} reviews for your selected application that were submitted within your specified date range.")
            else:
                st.write("Here's a sample of your scraped reviews:")
                st.dataframe(scrape_df.head())
                st.session_state["scrape_df"] = scrape_df
                st.session_state["scrape_n"] = scrape_df.shape[0]
                st.write(f"You have successfully scraped {scrape_df.shape[0]} reviews for your selected application.")
            
# # Exploratory Data Analysis
# if choice == "Exploratory Data Analysis":
#     st.header("Automated Exploratory Data Analysis")
#     if st.session_state["scrape_df"] is not None: 
#         report = st.session_state["scrape_df"].profile_report()
#         st.write("Current Application Scraped:" + " " + st.session_state["app_id"])
#         st_profile_report(report)
#     else:
#         st.warning("Please scrape reviews to get started.")  # Display a warning message if scrape_df is None


# Model Building for Sentiment Analysis            
if choice == "Sentiment Analysis":
    st.header("Sentiment Anaysis")
    st.write("NLP Model: Emotion English DistilRoBERTa-base ss")
    if st.session_state["scrape_df"] is not None:
            st.write("Current Application Scraped:" + " " + st.session_state["app_id"])
            data = st.session_state["scrape_df"]
            if st.button("Run Sentiment Analysis"):
                with st.spinner("Running Analysis..."):
                    emotions_means = {emotion: [] for emotion in emotions}
                    for index, row in data.iterrows():
                        review = row["review"]
                        date = row["date"]
                        prediction = classifier(str(review))
                        emotions_scores = {emotion: 0.0 for emotion in emotions}
                        if prediction and isinstance(prediction[0], list):
                            prediction_obj = prediction[0]
                            for emotion_score in prediction_obj:
                                emotion_label = emotion_score["label"]
                                emotion_score_value = emotion_score["score"]
                                if emotion_label in emotions:
                                    emotions_scores[emotion_label] = emotion_score_value
                                    emotions_means[emotion_label].append(emotion_score_value)
                        prediction_row = {"date": date, "review": review, **emotions_scores}
                        predictions.append(prediction_row)
                    predictions_df = pd.DataFrame(predictions)
                    st.write(f"Your current dataset contains a total of {st.session_state['scrape_n']} reviews.")
                    st.dataframe(predictions_df)

                    means = {emotion: np.mean(scores) for emotion, scores in emotions_means.items()}
                    # Calculate means for emotions
                    
                    means_data = {
                        "Emotion": list(emotions),
                        "Mean Emotion Score": [np.mean(scores) for emotion, scores in emotions_means.items()]
                    }

                    means_df = pd.DataFrame(means_data)

                    # Plot the bar chart with "Mean Emotion Score" as the y-axis label
                    st.bar_chart(means_df, x="Emotion", y="Mean Emotion Score")

    else:
        st.warning("Please scrape reviews to get started.")
