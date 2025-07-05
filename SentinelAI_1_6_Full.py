# SentinelAI 1.8 – מוח TorchMind חי, לומד פקודות מהצ'אט, מגיב רק כשמזהה
import torch
import torch.nn as nn
import socket
import threading
import subprocess
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# === עיבוד טקסט ===
def bag_of_words(tokenized_sentence, all_words):
    vector = torch.zeros(len(all_words), dtype=torch.float32)
    for word in tokenized_sentence:
        if word in all_words:
            index = all_words.index(word)
            vector[index] = 1.0
    return vector

# === TorchMind – רשת עצבית חיה ===
class TorchMind(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

# === ניהול הפקודות והמודל ===
class CommandAI:
    def __init__(self):
        self.commands = {}
        self.examples = []
        self.labels = []
        self.all_words = []
        self.label_map = {}
        self.reverse_map = {}
        self.model = None
        self.ready = False

    def add_example(self, text, label, js_code):
        tokens = word_tokenize(text.lower())
        self.examples.append(tokens)
        self.labels.append(label)
        self.commands[label] = js_code
        for token in tokens:
            if token not in self.all_words:
                self.all_words.append(token)
        if label not in self.label_map:
            idx = len(self.label_map)
            self.label_map[label] = idx
            self.reverse_map[idx] = label
        self.train()

    def train(self):
        if not self.examples:
            return
        X = [bag_of_words(x, self.all_words) for x in self.examples]
        y = torch.tensor([self.label_map[label] for label in self.labels])
        X = torch.stack(X)
        self.model = TorchMind(len(self.all_words), 64, len(self.label_map))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        for _ in range(300):
            out = self.model(X)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.ready = True

    def predict_js(self, message):
        if not self.ready:
            return None
        tokens = word_tokenize(message.lower())
        vec = bag_of_words(tokens, self.all_words)
        self.model.eval()
        with torch.no_grad():
            output = self.model(vec)
            probs = torch.softmax(output, dim=0)
            confidence, pred = torch.max(probs, 0)
            if confidence.item() < 0.75:
                return None
            label = self.reverse_map[pred.item()]
            return self.commands[label]

# === מייצר את קובץ הבוט ומתחבר למוח ===
def generate_bot_js():
    with open("bot.js", "w", encoding="utf-8") as f:
        f.write("""
const mineflayer = require('mineflayer');
const net = require('net');

const bot = mineflayer.createBot({
  host: 'localhost',
  port: 1234,
  username: 'SentinelAI',
  version: '1.19.4'
});

function askModel(text, callback) {
  const client = new net.Socket();
  client.connect(5050, 'localhost', () => {
    client.write(text);
  });
  client.on('data', (data) => {
    const code = data.toString().trim();
    if (code.length > 0) callback(code);
    client.destroy();
  });
  client.on('error', () => {});
}

bot.on('spawn', () => {
  console.log("🤖 SentinelAI התחבר לעולם!");
});

bot.on('chat', (username, message) => {
  if (username === bot.username) return;

  if (message.startsWith('!למד')) {
    const parts = message.split('|');
    if (parts.length === 4) {
      const learnMsg = JSON.stringify({
        learn: true,
        label: parts[1].trim(),
        text: parts[2].trim(),
        code: parts[3].trim()
      });
      const client = new net.Socket();
      client.connect(5050, 'localhost', () => {
        client.write(learnMsg);
        bot.chat(`✅ למדתי את הפקודה "${parts[1].trim()}"`);
        client.destroy();
      });
    } else {
      bot.chat("📘 שימוש: !למד | תגית | טקסט | קודJS");
    }
    return;
  }

  askModel(message, (jsCode) => {
    bot.chat(`🧠 מבצע: ${message}`);
    try {
      const vm = require('vm');
      const context = { bot, setTimeout, Vec3: require('vec3') };
      vm.createContext(context);
      vm.runInContext(jsCode, context);
    } catch (err) {
      console.log("⚠️ שגיאה בהרצה:", err.message);
    }
  });
});
""")

# === הרצה מלאה ===
def main():
    ai = CommandAI()

    # פקודות התחלתיות
    base_cmds = [
        ("לך קדימה", "go_forward", "bot.setControlState('forward', true);"),
        ("עצור", "stop", "bot.clearControlStates();"),
        ("קפוץ", "jump", "bot.setControlState('jump', true); setTimeout(() => bot.setControlState('jump', false), 300);"),
        ("תקוף", "attack", "const e = bot.nearestEntity(); if (e) bot.attack(e);")
    ]
    for text, label, js in base_cmds:
        ai.add_example(text, label, js)

    generate_bot_js()
    print("🚀 מריץ את הבוט...")
    subprocess.Popen(["node", "bot.js"])

    # שרת TorchMind
    def handle_client(conn):
        with conn:
            while True:
                data = conn.recv(2048)
                if not data:
                    break
                msg = data.decode('utf-8')
                if msg.startswith('{') and 'learn' in msg:
                    try:
                        info = eval(msg)
                        ai.add_example(info['text'], info['label'], info['code'])
                    except Exception as e:
                        print("❌ שגיאה בלימוד:", e)
                else:
                    js = ai.predict_js(msg)
                    if js:
                        conn.send(js.encode('utf-8'))
                    else:
                        conn.send("".encode('utf-8'))  # שקט מוחלט אם לא מזהה

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('localhost', 5050))
    server.listen()
    print("🧠 TorchMind מחכה לפקודות...")
    while True:
        conn, _ = server.accept()
        threading.Thread(target=handle_client, args=(conn,)).start()

if __name__ == "__main__":
    main()
