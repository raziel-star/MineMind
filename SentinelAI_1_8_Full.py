import os
import json
import subprocess
import threading
import torch
import torch.nn as nn
import torch.optim as optim
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)

# --- עיבוד טקסט ---
def bag_of_words(tokenized_sentence, all_words):
    word_to_index = {w: i for i, w in enumerate(all_words)}
    vector = torch.zeros(len(all_words), dtype=torch.float32)
    for w in tokenized_sentence:
        idx = word_to_index.get(w)
        if idx is not None:
            vector[idx] = 1.0
    return vector

# --- מודל TorchMind משודרג ---
class TorchMind(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

# --- ניהול AI ולמידה ---
class CommandAI:
    def __init__(self, data_file='learned_ai.json'):
        self.data_file = data_file
        self.commands = {}     # label -> JS code
        self.examples = []     # רשימת טוקניזציות
        self.labels = []       # תגיות
        self.all_words = []    # אוצר מילים
        self.label_map = {}    # label -> index
        self.reverse_map = {}  # index -> label
        self.model = None
        self.trained = False
        self.hidden_size = 128

        self.load_data()
        if len(self.labels) > 0:
            self.train()

    def add_example(self, text, label, js_code):
        tokens = word_tokenize(text.lower())
        self.examples.append(tokens)
        self.labels.append(label)
        self.commands[label] = js_code

        for t in tokens:
            if t not in self.all_words:
                self.all_words.append(t)

        if label not in self.label_map:
            idx = len(self.label_map)
            self.label_map[label] = idx
            self.reverse_map[idx] = label

    def train(self):
        if not self.examples:
            return
        X = torch.stack([bag_of_words(e, self.all_words) for e in self.examples])
        y = torch.tensor([self.label_map[l] for l in self.labels])

        input_size = len(self.all_words)
        output_size = len(self.label_map)

        self.model = TorchMind(input_size, self.hidden_size, output_size)
        optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(300):
            optimizer.zero_grad()
            output = self.model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        self.trained = True
        print(f"🧠 אימון הושלם - אוצר מילים: {input_size}, פקודות: {output_size}")

    def predict_js(self, text, threshold=0.7):
        if not self.trained:
            return None
        tokens = word_tokenize(text.lower())
        vec = bag_of_words(tokens, self.all_words)
        self.model.eval()
        with torch.no_grad():
            output = self.model(vec.unsqueeze(0))[0]
            probs = torch.softmax(output, dim=0)
            conf, pred = torch.max(probs, 0)
            if conf.item() < threshold:
                return None
            label = self.reverse_map[pred.item()]
            return self.commands.get(label, None)

    def save_data(self):
        data = {
            "commands": self.commands,
            "examples": [' '.join(e) for e in self.examples],
            "labels": self.labels,
            "all_words": self.all_words
        }
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("💾 שמירת נתוני למידה הסתיימה")

    def load_data(self):
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.commands = data.get("commands", {})
                self.labels = data.get("labels", [])
                self.examples = [e.split() for e in data.get("examples", [])]
                self.all_words = data.get("all_words", [])

                self.label_map = {}
                self.reverse_map = {}
                for label in set(self.labels):
                    idx = len(self.label_map)
                    self.label_map[label] = idx
                    self.reverse_map[idx] = label

# --- יצירת קובץ JS לבוט Mineflayer ---
def generate_bot_js():
    js = r"""
const mineflayer = require('mineflayer');
const vm = require('vm');
const readline = require('readline');

const bot = mineflayer.createBot({
  host: 'localhost',
  port: 1234,
  username: 'SentinelAI',
  version: '1.19.4'
});

bot.on('spawn', () => {
  console.log('[READY]');
  bot.chat('🤖 SentinelAI מחובר ומוכן!');
});

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

rl.on('line', (line) => {
  try {
    const context = { bot, setTimeout, Vec3: require('vec3'), console };
    vm.createContext(context);
    vm.runInContext(line, context);
  } catch(e) {
    console.log("JS ERROR:", e.message);
  }
});

bot.on('chat', (username, message) => {
  if(username === bot.username) return;
  console.log(`[CHAT]${username}:${message}`);
});
"""
    with open("bot.js", "w", encoding="utf-8") as f:
        f.write(js)

# --- הפעלת הבוט והרצת הלמידה בזמן אמת ---
def main():
    ai = CommandAI()

    # פקודות מובנות מגוונות
    base_cmds = [
        ("לך קדימה", "forward", "bot.setControlState('forward', true);"),
        ("תפס", "forward", "bot.setControlState('forward', true);"),
        ("לך אחורה", "back", "bot.setControlState('back', true);"),
        ("סגור", "stop", "bot.clearControlStates();"),
        ("עצור", "stop", "bot.clearControlStates();"),
        ("קפוץ", "jump", "bot.setControlState('jump', true); setTimeout(() => bot.setControlState('jump', false), 300);"),
        ("תקוף", "attack", "const e = bot.nearestEntity(); if(e) bot.attack(e);"),
        ("הסתכל למעלה", "look_up", "bot.look(bot.entity.yaw, bot.entity.pitch - 0.5, true);"),
        ("הסתכל למטה", "look_down", "bot.look(bot.entity.yaw, bot.entity.pitch + 0.5, true);"),
        ("הפעל יצירתיות", "creative_mode", "bot.chat('/gamemode creative');"),
        ("הפעל הישרדות", "survival_mode", "bot.chat('/gamemode survival');"),
        ("קח אש", "give_fire_charge", "bot.chat('/give @p minecraft:fire_charge 1');"),
        ("תן עץ", "give_wood", "bot.chat('/give @p minecraft:oak_log 64');"),
        ("תעוף", "fly", "bot.chat('/fly');"),
        ("הפסק לעוף", "stop_fly", "bot.chat('/fly off');"),
        ("פתח דלת", "open_door", """
const door = bot.blockAt(bot.entity.position.offset(0, 0, 1));
if(door && door.name.includes('door')) bot.activateBlock(door);
"""),
    ]

    # פקודות דינמיות - קפיצות עם מספרים שונים
    for i in range(1, 101):
        text = f"קפוץ {i} פעמים"
        label = f"jump_{i}"
        js = f"""
let count = 0;
function jumpLoop() {{
  if(count >= {i}) return;
  bot.setControlState('jump', true);
  setTimeout(() => {{
    bot.setControlState('jump', false);
    count++;
    setTimeout(jumpLoop, 300);
  }}, 300);
}}
jumpLoop();
"""
        ai.add_example(text, label, js)

    # הוספת כל הפקודות המובנות למודל
    for text, label, js in base_cmds:
        ai.add_example(text, label, js)

    ai.train()
    ai.save_data()
    generate_bot_js()

    print("🚀 מפעיל את הבוט ומתחיל להאזין לצ'אט...")

    proc = subprocess.Popen(
        ["node", "bot.js"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1
    )

    def listen():
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("[READY]"):
                print("✅ הבוט התחבר לעולם!")
            elif line.startswith("[CHAT]"):
                try:
                    user, msg = line[6:].split(":", 1)
                    msg = msg.strip()
                except Exception:
                    continue

                # פקודת לימוד חדשה !למד פקודה | טקסט | קוד JS
                if msg.startswith("!למד פקודה"):
                    parts = msg.split("|")
                    if len(parts) == 3:
                        text = parts[1].strip()
                        js_code = parts[2].strip()
                        label = text
                        print(f"[AI] לומד פקודה חדשה: {label}")
                        ai.add_example(text, label, js_code)
                        ai.train()
                        ai.save_data()
                        proc.stdin.write(f"bot.chat('✅ למדתי את הפקודה: {label}');\n")
                        proc.stdin.flush()
                    else:
                        proc.stdin.write("bot.chat('❌ פורמט שגוי ללימוד פקודה. השתמש: !למד פקודה | טקסט | קודJS');\n")
                        proc.stdin.flush()
                    continue

                # ניתוח פקודות רגילות
                js = ai.predict_js(msg)
                if js:
                    print(f"[AI] מבצע פקודה: {msg}")
                    proc.stdin.write(js + "\n")
                    proc.stdin.flush()
                    proc.stdin.write(f"bot.chat('🧠 מבצע: {msg}');\n")
                    proc.stdin.flush()
                else:
                    # לא מגיב בצ'אט אם לא מבין
                    print(f"[AI] לא זיהה פקודה: {msg}")

    thread = threading.Thread(target=listen, daemon=True)
    thread.start()

    # לולאת קלט ידנית למשתמש
    try:
        while True:
            inp = input(">>> ")
            if inp.strip() == "exit":
                print("⚠️ יציאה...")
                proc.terminate()
                break
            js = ai.predict_js(inp)
            if js:
                proc.stdin.write(js + "\n")
                proc.stdin.flush()
                print(f"[AI] ביצע פקודה ידנית: {inp}")
            else:
                print("[AI] לא זיהה פקודה.")
    except KeyboardInterrupt:
        proc.terminate()

if __name__ == "__main__":
    main()
