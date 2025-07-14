import os
import json
import subprocess
import threading
import torch
import torch.nn as nn
import torch.optim as optim
import nltk
from nltk.tokenize import word_tokenize
import time

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

# --- מודל TorchMind ---
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
        self.examples = []     # טוקניזציות
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
        # למנוע כפילויות
        if tokens in self.examples and label in self.labels:
            return
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

    def predict_js(self, text, threshold=0.9):
        if not self.trained:
            return None
        tokens = word_tokenize(text.lower())
        vec = bag_of_words(tokens, self.all_words)
        self.model.eval()
        with torch.no_grad():
            output = self.model(vec.unsqueeze(0))[0]
            probs = torch.softmax(output, dim=0)
            conf, pred = torch.max(probs, 0)

            print(f"[DEBUG] input: \"{text}\" | predicted: {self.reverse_map[pred.item()]} | confidence: {conf.item():.2f}")

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

    def retrain(self):
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
        for epoch in range(100):
            optimizer.zero_grad()
            output = self.model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        self.trained = True
        print(f"🧠 אימון מחודש הושלם - אוצר מילים: {input_size}, פקודות: {output_size}")

# --- יצירת קובץ JS לבוט Mineflayer עם DEBUG ופקודות חדשות ---

def generate_bot_js():
    js = r"""
const mineflayer = require('mineflayer');
const vm = require('vm');
const readline = require('readline');
const Vec3 = require('vec3');

const bot = mineflayer.createBot({
  username: 'Claude',
  version: '1.20.1',
  host: 'cuberazi.aternos.me',  // הכנס את ה-IP שלך כאן
  port: 25565,            // הכנס את הפורט שלך כאן
});

bot.once('spawn', () => {
  console.log('[DEBUG] EVENT: spawn');
  console.log('[READY]');
  bot.chat('🤖 Claude מחובר ומוכן! איך אני יכול לעזור לכם היום');
  startEnemyScan();
  startPatrol();
});

// אירועים בסיסיים
bot.on('login', () => {
  console.log('[DEBUG] EVENT: login');
});

bot.on('error', err => {
  console.error('[ERROR]', err.message);
  console.error('[ERROR FULL]', err);
});

bot.on('end', () => {
  console.log('[DISCONNECTED] הבוט נותק מהשרת');
});

bot.on('kicked', reason => {
  console.log('[KICKED] הבוט נדחה מהשרת:', reason);
});

bot.on('message', (message) => {
  console.log('[MESSAGE]', message.toString());
});

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

rl.on('line', (line) => {
  try {
    const context = {
      bot,
      require,
      console,
      setTimeout,
      clearTimeout,
      setInterval,
      clearInterval,
      Vec3
    };
    vm.createContext(context);
    vm.runInContext(line, context);
  } catch(e) {
    console.error("[JS ERROR]:", e.message);
  }
});

bot.on('chat', (username, message) => {
  if(username === bot.username) return;
  console.log(`[CHAT]${username}:${message}`);
});

// פקודות בסיסיות
const commands = {
  "forward": () => bot.setControlState('forward', true),
  "back": () => bot.setControlState('back', true),
  "stop": () => bot.clearControlStates(),
  "jump": () => {
    bot.setControlState('jump', true);
    setTimeout(() => bot.setControlState('jump', false), 300);
  },
  "attack": () => {
    const e = bot.nearestEntity(e => e.type === 'mob' && e.position.distanceTo(bot.entity.position) < 10);
    if(e) bot.attack(e);
  },
  "look_up": () => bot.look(bot.entity.yaw, bot.entity.pitch - 0.5, true),
  "look_down": () => bot.look(bot.entity.yaw, bot.entity.pitch + 0.5, true),
  "creative_mode": () => bot.chat('/gamemode creative'),
  "survival_mode": () => bot.chat('/gamemode survival'),
  "give_fire_charge": () => bot.chat('/give @p minecraft:fire_charge 1'),
  "give_wood": () => bot.chat('/give @p minecraft:oak_log 64'),
  "fly": () => bot.chat('/fly'),
  "stop_fly": () => bot.chat('/fly off'),
  "open_door": () => {
    const door = bot.blockAt(bot.entity.position.offset(0, 0, 1));
    if(door && door.name.includes('door')) bot.activateBlock(door);
  },
  "dig_down": () => {
    const blockBelow = bot.blockAt(bot.entity.position.offset(0, -1, 0));
    if (blockBelow) bot.dig(blockBelow);
  },
  "dig_forward": () => {
    const vec = bot.entity.position.offset(0, 0, 1);
    const block = bot.blockAt(vec);
    if (block) bot.dig(block);
  },
  "collect_nearby_items": () => {
    const items = bot.entities;
    for (const id in items) {
      const entity = items[id];
      if (entity.entityType === 2 && bot.entity.position.distanceTo(entity.position) < 5) {
        bot.collectBlock.collect(entity);
      }
    }
  },
  "look_around": () => {
    let yaw = bot.entity.yaw;
    const interval = setInterval(() => {
      yaw += Math.PI / 4;
      bot.look(yaw, bot.entity.pitch, true);
    }, 500);
    setTimeout(() => clearInterval(interval), 4000);
  },
  "find_nearest_entity": () => {
    const e = bot.nearestEntity();
    if(e) bot.chat('מצאתי יצור: ' + (e.username || e.name));
    else bot.chat('לא מצאתי יצור קרוב');
  },
  "give_torch": () => bot.chat('/give @p torch 64'),
  "open_door_forward": () => {
    const door = bot.blockAt(bot.entity.position.offset(0, 0, 1));
    if (door && door.name.includes('door')) bot.activateBlock(door);
  },
  "build_wood_wall": async () => {
    const pos = bot.entity.position.offset(1, 0, 0);
    for(let i=0; i<5; i++) {
      const blockPos = pos.offset(i, 0, 0);
      const block = bot.blockAt(blockPos);
      if (block && block.name === 'air') {
        await bot.placeBlock(bot.blockAt(blockPos.offset(0, -1, 0)), new Vec3(0, 1, 0));
      }
    }
  },
  "dance": () => {
    bot.setControlState('left', true);
    setTimeout(() => bot.setControlState('left', false), 500);
    setTimeout(() => bot.setControlState('right', true), 600);
    setTimeout(() => bot.setControlState('right', false), 1100);
  },
};

// פקודות דינמיות - קפוץ מספר פעמים
for(let i=1; i<=100; i++) {
  commands[`jump_${i}`] = () => {
    let count = 0;
    function jumpLoop() {
      if(count >= i) return;
      bot.setControlState('jump', true);
      setTimeout(() => {
        bot.setControlState('jump', false);
        count++;
        setTimeout(jumpLoop, 300);
      }, 300);
    }
    jumpLoop();
  }
}

// פקודת תקיפת זומבים ספציפית
commands["attack_zombie"] = () => {
  const zombie = bot.nearestEntity(e => e.type === 'mob' && e.mobType === 'Zombie' && e.position.distanceTo(bot.entity.position) < 16);
  if(zombie) {
    bot.chat('זומבי איתרתי! תוקף עכשיו!');
    bot.attack(zombie);
  } else {
    bot.chat('לא מצאתי זומבי בסביבה.');
  }
};

// סריקת ברזל ותחילת חציבה אוטומטית
function startIronMineScan() {
  setInterval(() => {
    const pos = bot.entity.position;
    const range = 10;
    for(let dx=-range; dx<=range; dx++) {
      for(let dy=-2; dy<=2; dy++) {
        for(let dz=-range; dz<=range; dz++) {
          const blockPos = pos.offset(dx, dy, dz);
          const block = bot.blockAt(blockPos);
          if(block && block.name.includes('iron_ore')) {
            bot.dig(block);
            return;
          }
        }
      }
    }
  }, 10000);
}

// סריקה מתמדת לסיור וחיפוש אויבים
function startEnemyScan() {
  setInterval(() => {
    const enemy = bot.nearestEntity(e => e.type === 'zombie' && e.position.distanceTo(bot.entity.position) < 16);
    if(enemy) {
      bot.attack(enemy);
    }
  }, 5000);
}

// סיור אוטומטי במרחב מסוים
function startPatrol() {
  let yaw = bot.entity.yaw;
  setInterval(() => {
    yaw += Math.PI / 4;
    bot.look(yaw, bot.entity.pitch, true);
  }, 4000);
}

// הפעלת פקודות שהגיעו מה-Python
function executeCommand(label) {
  const cmd = commands[label];
  if(cmd) cmd();
}

module.exports = {
  executeCommand,
  commands
};
"""
    with open("bot.js", "w", encoding="utf-8") as f:
        f.write(js)


# --- הפעלת הבוט והרצת הלמידה בזמן אמת ---
def main():
    ai = CommandAI()

    # פקודות מובנות Minecraft (כולל פקודות חדשות)
    base_cmds = [
        ("לך קדימה", "forward", "bot.setControlState('forward', true);"),
        ("תפס", "forward", "bot.setControlState('forward', true);"),
        ("לך אחורה", "back", "bot.setControlState('back', true);"),
        ("סגור", "stop", "bot.clearControlStates();"),
        ("עצור", "stop", "bot.clearControlStates();"),
        ("קפוץ", "jump", "bot.setControlState('jump', true); setTimeout(() => bot.setControlState('jump', false), 300);"),
        ("תקוף", "attack", """
const e = bot.nearestEntity(e => e.type === 'mob' && e.position.distanceTo(bot.entity.position) < 10);
if(e) bot.attack(e);
"""),
        ("תקוף זומבי", "attack_zombie", """
const zombie = bot.nearestEntity(e => e.type === 'mob' && e.mobType === 'Zombie' && e.position.distanceTo(bot.entity.position) < 16);
if(zombie) {
  bot.chat('זומבי איתרתי! תוקף עכשיו!');
  bot.attack(zombie);
} else {
  bot.chat('לא מצאתי זומבי בסביבה.');
}
"""),
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
        ("חפור למטה", "dig_down", """
const blockBelow = bot.blockAt(bot.entity.position.offset(0, -1, 0));
if (blockBelow) bot.dig(blockBelow);
"""),
        ("חפור קדימה", "dig_forward", """
const vec = bot.entity.position.offset(0, 0, 1);
const block = bot.blockAt(vec);
if (block) bot.dig(block);
"""),
        ("אסוף את כל הסביבה", "collect_nearby_items", """
const items = bot.entities;
for (const id in items) {
  const entity = items[id];
  if (entity.entityType === 2 && bot.entity.position.distanceTo(entity.position) < 5) {
    bot.collectBlock.collect(entity);
  }
}
"""),
        ("הסתכל סביב", "look_around", """
let yaw = bot.entity.yaw;
const interval = setInterval(function () {
  yaw += Math.PI / 4;
  bot.look(yaw, bot.entity.pitch, true);
}, 500);
globalThis._lookAroundTimer = setTimeout(() => {
  clearInterval(interval);
}, 4000);
"""),
        ("מצא יצור קרוב", "find_nearest_entity", """
const e = bot.nearestEntity();
if(e) bot.chat('מצאתי יצור: ' + (e.username || e.name));
else bot.chat('לא מצאתי יצור קרוב');
"""),
        ("הדליק לפיד", "give_torch", "bot.chat('/give @p torch 64');"),
        ("פתח דלת לפנים", "open_door_forward", """
const door = bot.blockAt(bot.entity.position.offset(0, 0, 1));
if (door && door.name.includes('door')) bot.activateBlock(door);
"""),
        ("בנה קיר עץ", "build_wood_wall", """
const Vec3 = require('vec3');
const pos = bot.entity.position.offset(1, 0, 0);
async function buildWall() {
  for(let i=0; i<5; i++) {
    const blockPos = pos.offset(i, 0, 0);
    const block = bot.blockAt(blockPos);
    if (block && block.name === 'air') {
      await bot.placeBlock(bot.blockAt(blockPos.offset(0, -1, 0)), new Vec3(0, 1, 0));
    }
  }
}
buildWall();
"""),
        ("רקוד", "dance", """
bot.setControlState('left', true);
setTimeout(() => bot.setControlState('left', false), 500);
setTimeout(() => bot.setControlState('right', true), 600);
setTimeout(() => bot.setControlState('right', false), 1100);
"""),
    ]

    # פקודות דינמיות - קפוץ מספר פעמים
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

    # הוספת פקודות מובנות
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

    connected_flag = False

    def listen():
        nonlocal connected_flag
        for line in proc.stdout:
            line = line.strip()
            print(f"[BOT LOG] {line}")
            if line.startswith("[READY]"):
                connected_flag = True
                print("✅ הבוט התחבר לעולם! (Detected in Python)")
            elif line.startswith("[CHAT]"):
                try:
                    user, msg = line[6:].split(":", 1)
                    msg = msg.strip()
                    print(f"[CHAT RECEIVED] {user}: {msg}")
                except Exception:
                    continue

                # פקודת לימוד חדשה, שמתחילה ב-!למד פקודה
                if msg.startswith("!למד פקודה"):
                    parts = msg.split("|")
                    if len(parts) == 3:
                        text = parts[1].strip()
                        js_code = parts[2].strip()
                        label = text
                        print(f"[AI] לומד פקודה חדשה: {label}")
                        ai.add_example(text, label, js_code)
                        ai.retrain()
                        ai.save_data()
                        proc.stdin.write(f"bot.chat('✅ למדתי את הפקודה: {label}');\n")
                        proc.stdin.flush()
                    else:
                        proc.stdin.write("bot.chat('❌ פורמט שגוי ללימוד פקודה. השתמש: !למד פקודה | טקסט | קודJS');\n")
                        proc.stdin.flush()
                    continue

                if msg.startswith("!"):
                    # הודעה שמתחילה בסימן קריאה אבל לא לימוד
                    js = ai.predict_js(msg[1:])
                    if js:
                        # פקודה מוכרת - מבצעים ועונים
                        print(f"[AI] מבצע פקודה: {msg}")
                        proc.stdin.write(js + "\n")
                        proc.stdin.flush()
                        proc.stdin.write(f"bot.chat('🧠 מבצע: {msg}');\n")
                        proc.stdin.flush()
                    else:
                        # פקודה לא מוכרת עם סימן קריאה - לא עונים ולא מבצעים
                        print(f"[AI] פקודה לא מוכרת עם סימן קריאה - לא עונה: {msg}")
                    continue  # ממשיכים ללולאה בלי תגובה

                else:
                    # הודעה שלא מתחילה בסימן קריאה
                    js = ai.predict_js(msg)
                    if js:
                        # פקודה מוכרת - מבצעים אך לא עונים
                        print(f"[AI] מבצע פקודה (בלי תגובה בצאט): {msg}")
                        proc.stdin.write(js + "\n")
                        proc.stdin.flush()
                    else:
                        # הודעה רגילה שלא מזוהה - לא עונים
                        print(f"[AI] הודעה רגילה שלא מזוהתה - לא עונה: {msg}")

    thread = threading.Thread(target=listen, daemon=True)
    thread.start()

    # timeout התחברות אחרי 15 שניות
    timeout = 15
    start_time = time.time()
    while time.time() - start_time < timeout:
        if connected_flag:
            break
        time.sleep(0.1)

    if not connected_flag:
        print("❌ הבוט לא התחבר תוך 15 שניות, בדוק את פרטי החיבור ושרת המשחק.")

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
