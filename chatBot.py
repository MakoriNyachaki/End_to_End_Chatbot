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
 
