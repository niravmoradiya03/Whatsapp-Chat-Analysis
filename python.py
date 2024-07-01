import streamlit as st
import re
import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
from wordcloud import WordCloud
import streamlit as st

def main():
    # Set background image URL
    background_image = 'https://www.teleosms.com/wp-content/uploads/2021/09/whatsapp-bg.jpg'

    # Define CSS style with background image
    page_bg_img = '''
    <style>
    body {
    background-image: url("%s");
    background-size: cover;
    }
    </style>
    ''' % background_image

    # Inject CSS into Streamlit app
    st.markdown(page_bg_img, unsafe_allow_html=True)

  

if __name__ == '__main__':
    main()


def time_to_minutes(time, am_pm):
    hours, minutes = map(int, time.split(':'))
    total_minutes = hours * 60 + minutes
    if am_pm.lower() == 'pm' and hours != 12:
        total_minutes += 720
    elif am_pm.lower() == 'am' and hours == 12:
        total_minutes -= 720
    return total_minutes

def contains_url(message):
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return bool(url_pattern.search(message))

st.title('Whatsapp Chat Analysis')
uploaded_file = st.file_uploader("Choose a text file", type="txt")

if uploaded_file is not None:
    raw = []
    pattern = r"(\d{2}/\d{2}/\d{2}), (\d{1,2}:\d{2}\s[apAP][mM]) - (.+)"
    file_content = StringIO(uploaded_file.getvalue().decode("utf-8"))
    
    for line in file_content.readlines():
        x = re.search(pattern, line)
        if x is None:
            if len(raw) > 0:
                raw[-1][-1] += ' ' + line.strip()
        else:
            raw.append([*x.groups()])
    
    output = StringIO()
    csv_writer = csv.writer(output)
    csv_writer.writerow(['date', 'time', 'am/pm', 'name', 'message'])
    for line in raw:
        date, time_am_pm, message = line
        time, am_pm = time_am_pm.split()
        name_message_split = message.split(':', 1)
        name = name_message_split[0] if len(name_message_split) == 2 else ""
        message = name_message_split[1].strip() if len(name_message_split) == 2 else name_message_split[0]
        csv_writer.writerow([date, time, am_pm, name.strip(), message.strip()])
    output.seek(0)
    df_chat = pd.read_csv(output)
    df_chat.rename(columns={'am/pm': 'meridiem'}, inplace=True)
    df_chat.dropna(inplace=True)
    df_chat['time_in_min'] = df_chat.apply(lambda row: time_to_minutes(row['time'], row['meridiem']), axis=1)
    df_chat['rounded_hours'] = (df_chat['time_in_min'] / 60).round()

    df_chat['contains_link'] = df_chat['message'].apply(contains_url)
    num_messages_with_links = df_chat['contains_link'].sum()
    st.write(f"Number of messages with links: {num_messages_with_links}")


    num_categories = len(df_chat['name'].value_counts())

    fig, ax = plt.subplots(figsize=(12, 8))
    df_chat['name'].value_counts().head(25).plot(kind="bar", ax=ax,)
    ax.set_title('Top 25 Most Active Chat Participants')
    ax.set_xlabel('Name')
    ax.set_ylabel('Message Count')
    st.pyplot(fig)

    num_messages = len(df_chat['message'].value_counts())

    fig, ax = plt.subplots(figsize=(12, 8))
    df_chat['message'].value_counts().head(3).plot(kind="bar", ax=ax, )
    ax.set_title('Top 3 Most Common Messages')
    ax.set_xlabel('Messages')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(12, 8))
    df_chat['hour_category'] = pd.cut(df_chat['time_in_min'], bins=24, labels=range(24))
    df_chat['hour_category'].value_counts().head(24).plot(kind="bar", ax=ax, )
    ax.set_title('Top 24 Hour Categories')
    ax.set_xlabel('Hour Category')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    # Combine all the messages into a single string
    text = " ".join(message for message in df_chat.message)

    # Create the word cloud object
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Plot the Word Cloud
    st.subheader("Word Cloud of Messages")
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Assuming 'message' is the name of the column you want to perform word count on
    column_to_count = df_chat['message']

    # Convert NaN values to empty strings and then join all the text in the column into a single string
    text = ' '.join(column_to_count.fillna('').astype(str))

    # Split the text into words and get total word count
    total_word_count = len(text.split())

    # Print the total word count
    st.write("Total word count:", total_word_count)

    # Download the updated dataset streamlit run streamlit.py --server.enableCORS false --server.enableXsrfProtection false
    updated_output = StringIO()
    df_chat.to_csv(updated_output, index=False)
    updated_output.seek(0)
    st.download_button(
        label="Download CSV",
        data=updated_output.getvalue(),
        file_name='updated_dataset.csv',
        mime='text/csv',
    )