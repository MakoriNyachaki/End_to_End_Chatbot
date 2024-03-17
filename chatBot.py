import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os)

# Defining intents

intents = [
	{
		"tag": "greeting"
		"responses": ["Hi there", "Hello", "Hey", "I'm fine, thank you", "Nothing much"]
		"patterns": ["Hi", "Hello", "How are you", "What's up"]
	},
	{
		"tag": "goodbye",
		"patterns": ["Bye", "See you later", "Goodbye", "Take care"],
		"responses": ["Goodbye", "See you later", "Take care"]
	},
	{
		"tag": "thanks",
		"patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
		"responses": ["You're welcome", "No problem", "Glad I could help"]
	},
	{
		"tag": "about",
		"patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
		"responses": ["I am a chatbot", "My purpose is to assist you", "I can answer questions and provide assistance"]
	},
	{
		"tag": "help",
		"patterns": ["Help", "I need help", "Can you help me", "What can I do"],
		"responses": ["sure what do you need help with?", "I'm here to help, what is the problem?", "How can I assist you?"]
	},
	{
		"tag": "age",
		"patterns": ["How old are you", "What's your age"],
		"responses": ["I don't have an age, I am a chatbot.", "I was just born in the digital world.", "Age is just a number for me."]
	},
	{
		"tag": "weather",
		"patterns": ["What's the weather like", "How is the weather today"],
		"responses": ["I'm sorry, I cannot provide realtime weather information", "You can check the weather on a weather app or website."]
	},
	{
		"tag": "budget",
		"patterns": ["How can I make a budget", "What's a good budgeting startegy", "How do I create a budget"],
		"responses": ["To make a budget start by tracking your income and expenses. Then, allocate your income towards essential expenses like rent, food, and bills. Next, alocate some of your income towards savings and debt repayment. Finally, allocate the remainder of your income towards discretionary expenses like entertainment and hobbies.", "A good budget strategy is to use 50/30/20 rule. This measn allocating 50% of your income towards essential expenses, 30% towards discretionary ex;penses, and 20% towards savings and debt repayment.", "To create a budget start by setting financial goals for yourself. Then, track your income and expenses for a few months to get a sense of where your money is going. Next, create a budget by allocating your income towards essential expenses, savings and debt repayment, and discretinary expenses."]
	},
	{
		"tag": "credit_score",
		"patterns": ["What is a credit score", "How do i check my credit score", "How can I improve my credit score"],
		"responses": []
	
	}

# Create the vectorizer and the classfier

vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=1000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
	for pattern in intent['patterns']:
		tags.append(intent['tag'])
		patterns.append(pattern)

# training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)



# Function to chat with the chatbot

def chatbot(input_text):
	input_text = vectorizer.transform([input_text])
	tag = clf.predict(input_text)[0]
	for intent in intents:
		if intent['tag'] == tag:
			response = random.choice(intent['responses'])
			return response

# Main function
counter = 0

def main():
	global counter
	st.title("ChatBot")
	st.write("Welcome to the chatbot. Please type a message and press enter to start the conversation.")

	counter += 1
	user_input = st.text_input("You:", key=f"user_input_{counter}")
	if user_input:
		response = chatbot(user_input)
		st.text_area("Chatbot:", value=response, height=100, max_chars=None, key=f"chatbot_response_{counter}")
		if response.lower() in ['goodbye', 'bye']:
			st.write("Thank you for chatting with me. Have a great day!")
			st.stop()


if __name__ == '__main__':
	main()
