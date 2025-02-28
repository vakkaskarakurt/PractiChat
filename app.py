from flask import Flask, request, render_template, jsonify
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import nltk
import pandas as pd
import re
import math

app = Flask(__name__)

# Load the model and tokenizer
model_dir = 'DialoGPT_Model'
tokenizer_dir = 'DialoGPT_Tokenizer'

tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_dir, padding_side='left')
model = GPT2LMHeadModel.from_pretrained(model_dir)

# Move the model to the CPU or GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Ensure nltk punkt tokenizer is downloaded
nltk.download('punkt')

# TextScorer class
class TextScorer:
    def __init__(self, csv_file):
        self.word_freq = self._load_word_frequencies(csv_file)

    def _load_word_frequencies(self, csv_file):
        df = pd.read_csv(csv_file)
        return dict(zip(df['word'], df['index']))

    def _clean_text(self, text):
        return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s]', '', text.lower())).strip()

    def _calculate_score(self, words):
        return sum(math.log(self.word_freq.get(word, 0) + 1, 5000) if word in self.word_freq else 1 for word in words) / len(words)

    def _adjust_score(self, score, word_count):
        length_score = math.log(word_count + 1, 15)
        return round(score * length_score * 100, 2)

    def evaluate_text(self, text):
        words = self._clean_text(text).split()
        return self._adjust_score(self._calculate_score(words), len(words))

# Initialize TextScorer
text_scorer = TextScorer('unigram_freq_index.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']

    if user_input.lower() in ['bye']:
        return jsonify({'response': "Goodbye!", 'user_input_score': text_scorer.evaluate_text("bye"), 'bot_response_score': text_scorer.evaluate_text("Goodbye!")})
    # Special case for "hello"
    if user_input.lower() == "hello":
        return jsonify({'response': "Hello! How can I help you today?", 'user_input_score': text_scorer.evaluate_text("hello"), 'bot_response_score': text_scorer.evaluate_text("Hello! How can I help you today?")})

    # Encode the user input
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt').to(device)

    # Generate a response without chat history
    chat_history_ids = model.generate(new_user_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode and get the bot's response
    bot_response = tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Use nltk for better sentence splitting
    sentences = nltk.sent_tokenize(bot_response)
    answer = sentences[0]  # Take the first sentence

    # Evaluate the text scores
    user_input_score = text_scorer.evaluate_text(user_input)
    bot_response_score = text_scorer.evaluate_text(answer)

    response = {
        'response': f"{answer}",
        'user_input_score': user_input_score,
        'bot_response_score': bot_response_score
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
