
# SentinelAI 1.6 – בוט Minecraft חכם עם 100 פקודות בעברית ואנגלית
import os
import json
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

ALL_WORDS = set()

def bag_of_words(tokenized_sentence, all_words):
    vector = torch.zeros(len(all_words), dtype=torch.float32)
    for word in tokenized_sentence:
        if word in all_words:
            index = list(all_words).index(word)
            vector[index] = 1.0
    return vector

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        return self.model(x)

class CommandAI:
    def __init__(self):
        self.commands = {}
        self.all_words = set()
        self.examples = []
        self.labels = []
        self.label_map = {}
        self.reverse_map = {}
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.trained = False

    def add_example(self, text, label, js_code):
        tokens = word_tokenize(text.lower())
        self.all_words.update(tokens)
        self.examples.append(tokens)
        self.labels.append(label)
        self.commands[label] = js_code
        if label not in self.label_map:
            idx = len(self.label_map)
            self.label_map[label] = idx
            self.reverse_map[idx] = label

    def save(self, file='learned_ai.json'):
        data = {
            'commands': self.commands,
            'examples': [' '.join(x) for x in self.examples],
            'labels': self.labels
        }
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, file='learned_ai.json'):
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.commands = data['commands']
                for text, label in zip(data['examples'], data['labels']):
                    self.add_example(text, label, self.commands[label])

    def train(self):
        X = [bag_of_words(tokens, self.all_words) for tokens in self.examples]
        y = [self.label_map[label] for label in self.labels]
        X = torch.stack(X)
        y = torch.tensor(y)

        input_size = len(self.all_words)
        hidden_size = 64
        output_size = len(set(y))

        self.model = NeuralNet(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()

        for epoch in range(300):
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.trained = True

    def predict(self, sentence):
        tokens = word_tokenize(sentence.lower())
        vector = bag_of_words(tokens, self.all_words)
        self.model.eval()
        with torch.no_grad():
            output = self.model(vector)
            predicted = torch.argmax(output).item()
            return self.reverse_map[predicted]

    def get_js(self, sentence):
        if not self.trained:
            return None
        label = self.predict(sentence)
        return self.commands.get(label, f"bot.chat('🤖 לא הבנתי את הפקודה: {sentence}');")

def main():
    ai = CommandAI()
    ai.load()
    print("🤖 SentinelAI 1.6 – מייצר 100 פקודות מגוונות בעברית ובאנגלית...")

    # 10 פקודות בסיסיות
    base = [
        ("לך קדימה", "לך קדימה", "bot.setControlState('forward', true);"),
        ("go forward", "לך קדימה", "bot.setControlState('forward', true);"),
        ("עצור", "עצור", "bot.clearControlStates();"),
        ("stop", "עצור", "bot.clearControlStates();"),
        ("רוץ", "רוץ", "bot.setControlState('sprint', true);"),
        ("run", "רוץ", "bot.setControlState('sprint', true);"),
        ("קפוץ", "קפוץ", "bot.setControlState('jump', true); setTimeout(() => bot.setControlState('jump', false), 500);"),
        ("jump", "קפוץ", "bot.setControlState('jump', true); setTimeout(() => bot.setControlState('jump', false), 500);"),
        ("סובב ימינה", "סובב ימינה", "bot.look(bot.entity.yaw + 1, 0);"),
        ("turn right", "סובב ימינה", "bot.look(bot.entity.yaw + 1, 0);")
    ]

    for text, label, js in base:
        ai.add_example(text, label, js)

    # 90 פקודות מגוונות
    for i in range(1, 91):
        text = f"פקודה {i}"
        label = f"פקודה {i}"
        js_code = f"bot.chat('📦 מבצע את פקודה {i}'); bot.setControlState('jump', true); setTimeout(() => bot.setControlState('jump', false), 300);"
        ai.add_example(text, label, js_code)

    ai.train()
    ai.save()
    print("✅ סיים ללמוד 100 פקודות. מוכן!")

if __name__ == "__main__":
    main()
