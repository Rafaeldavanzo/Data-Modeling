import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
from collections import Counter
import spacy

# Load the spacy model
nlp = spacy.load('en_core_web_sm')

# Function to remove personal names using spacy NER
def remove_personal_names(text):
    doc = nlp(text)
    cleaned_text = ' '.join([token.text for token in doc if token.ent_type_ != 'PERSON'])
    return cleaned_text

# Step 1: Read the CSV file with robust error handling
file_path = 'C:/Users/XXXXXXX/Documents/ticket_exports_20_04_24_TO_20_06_24.csv'
try:
    tickets = pd.read_csv(file_path, delimiter=',', quotechar='"', escapechar='\\', on_bad_lines='skip', engine='python')
except pd.errors.ParserError as e:
    print(f"Error reading the CSV file: {e}")

# Step 2: Display basic information about the dataset
print("Basic Information about the dataset:")
print(tickets.info())

print("\nFirst few rows of the dataset:")
print(tickets.head())

# Display column names to understand the dataset structure
print("\nColumn names in the dataset:")
print(tickets.columns)

# Step 3: Clean and preprocess the data (if necessary)
# Check for missing values
print("\nMissing values in each column:")
print(tickets.isnull().sum())

# If there are any columns with significant missing values, we may decide to drop or fill them
# For example, filling missing values in 'priority' with 'Unknown' if it exists
if 'priority' in tickets.columns:
    tickets['priority'].fillna('Unknown', inplace=True)

# Convert 'created_at' and 'updated_at' to datetime format if they exist
def safe_to_datetime(column):
    try:
        return pd.to_datetime(column)
    except ValueError:
        return pd.to_datetime(column, errors='coerce')

if 'created_at' in tickets.columns:
    tickets['created_at'] = safe_to_datetime(tickets['created_at'])
if 'updated_at' in tickets.columns:
    tickets['updated_at'] = safe_to_datetime(tickets['updated_at'])

# Drop rows where datetime conversion failed
tickets.dropna(subset=['created_at', 'updated_at'], inplace=True)

# Remove personal names from the 'description' field
if 'description' in tickets.columns:
    tickets['description'] = tickets['description'].apply(remove_personal_names)

# Step 4: Perform exploratory data analysis (EDA)
# Let's start with some basic analysis if the columns exist

# Number of tickets by status
if 'status' in tickets.columns:
    status_counts = tickets['status'].value_counts()
    print("\nNumber of tickets by status:")
    print(status_counts)
    
    # Plot number of tickets by status
    plt.figure(figsize=(10, 6))
    sns.countplot(data=tickets, x='status', order=status_counts.index)
    plt.title('Number of Tickets by Status')
    plt.xlabel('Status')
    plt.ylabel('Count')
    plt.show()

# Number of tickets by priority
if 'priority' in tickets.columns:
    priority_counts = tickets['priority'].value_counts()
    print("\nNumber of tickets by priority:")
    print(priority_counts)
    
    # Plot number of tickets by priority
    plt.figure(figsize=(10, 6))
    sns.countplot(data=tickets, x='priority', order=priority_counts.index)
    plt.title('Number of Tickets by Priority')
    plt.xlabel('Priority')
    plt.ylabel('Count')
    plt.show()

# Number of tickets by type
if 'type' in tickets.columns:
    type_counts = tickets['type'].value_counts()
    print("\nNumber of tickets by type:")
    print(type_counts)
    
    # Plot number of tickets by type
    plt.figure(figsize=(10, 6))
    sns.countplot(data=tickets, x='type', order=type_counts.index)
    plt.title('Number of Tickets by Type')
    plt.xlabel('Type')
    plt.ylabel('Count')
    plt.show()

# Tickets created over time
if 'created_at' in tickets.columns:
    tickets['month_year'] = tickets['created_at'].dt.to_period('M')
    tickets_by_month = tickets.groupby('month_year').size()
    print("\nTickets created over time:")
    print(tickets_by_month)
    
    # Plot tickets created over time
    plt.figure(figsize=(12, 6))
    tickets_by_month.plot(kind='bar')
    plt.title('Tickets Created Over Time')
    plt.xlabel('Month-Year')
    plt.ylabel('Number of Tickets')
    plt.xticks(rotation=45)
    plt.show()

# Step 5: Generate a word cloud for the 'description' field
if 'description' in tickets.columns:
    # Combine all descriptions into one text
    text = " ".join(description for description in tickets.description.dropna())
    
    # Define stopwords and add additional words to exclude
    stopwords = set(STOPWORDS)
    additional_stopwords = {'fenergo','regarded','confidential','ms','the', 'in', 'to', 'this', 'you', 'and', 'by', 'of', 'be', 'from', 'X','destroy','error','become',}
    stopwords.update(additional_stopwords)
    
    # Generate the word cloud
    wordcloud = WordCloud(stopwords=stopwords, background_color='white', width=800, height=400).generate(text)
    
    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Ticket Descriptions')
    plt.show()
    
    # Print the 50 most mentioned words (excluding stopwords)
    words = [word for word in text.split() if word.lower() not in stopwords]
    word_counts = Counter(words)
    most_common_words = word_counts.most_common(50)
    print("\n50 Most Mentioned Words:")
    for word, count in most_common_words:
        print(f"{word}: {count}")

# Step 6: Sentiment Analysis
if 'description' in tickets.columns:
    # Function to calculate sentiment
    def get_sentiment(text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    
    # Apply sentiment analysis to the 'description' column
    tickets['sentiment'] = tickets['description'].dropna().apply(get_sentiment)
    
    # Classify sentiment
    tickets['sentiment_label'] = tickets['sentiment'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))
    
    # Plot sentiment distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=tickets, x='sentiment_label', order=['positive', 'neutral', 'negative'])
    plt.title('Sentiment Analysis of Ticket Descriptions')
    plt.xlabel('Sentiment')
    plt.ylabel('Frequency')
    plt.show()
    
    # Display the sentiment distribution
    sentiment_counts = tickets['sentiment_label'].value_counts()
    print("\nSentiment distribution in ticket descriptions:")
    print(sentiment_counts)
    
    # Create a stacked bar chart for sentiment distribution
    plt.figure(figsize=(10, 6))
    sentiment_counts.plot(kind='bar', color=['green', 'gray', 'red'])
    plt.title('Sentiment Distribution in Ticket Descriptions')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()
