# SentinelAI 1.6 – Smart Minecraft Bot with Natural Language Commands

**SentinelAI** is a smart AI bot for Minecraft that understands and responds to **natural language commands** in both **English and Hebrew**. It uses real machine learning (PyTorch), NLP, and connects with `mineflayer` to control your bot in real time.

---

## 📦 What's Inside

- `SentinelAI_1_6.py` – Python AI code (NLP + ML).
- `bot.js` – Minecraft bot using Mineflayer.
- `learned_ai.json` – Trained commands with examples.
- `README.md` – This guide.

---

## 🧠 Features

- Understands over **100 smart commands** like:
  - `Go forward`, `Stop`, `Attack`, `Jump 5 times`, `Build block`, `Eat`, `Sleep`, `Mine block`, `Patrol`, and more.
- Works in **real-time** with your Minecraft server.
- Learns new commands using `!למד פקודה` in the chat.
- Supports **Hebrew and English**.
- Built with **PyTorch + NLTK + Mineflayer**.

---

## 🚀 Requirements

Before running SentinelAI, make sure you have the following:

### 🐍 Python Side

- Python 3.9 or higher
- Install dependencies:
  ```bash
  pip install torch nltk
  ```

### 🟨 Node.js Side

- Node.js (v16 or higher recommended)
- Install dependencies in the folder:
  ```bash
  npm install mineflayer
  ```

---

## 🔧 How to Run

### 1. Launch the AI Engine (Python)

Open terminal and run:

```bash
python SentinelAI_1_6.py
```

This will train/load the model, generate `bot.js`, and run the bot.

---

### 2. Minecraft Server

Make sure your Minecraft server is running (e.g. on `localhost:25565`)  
The bot will connect as player `SentinelAI`.

---

## 💬 Using the Bot

In Minecraft chat, type commands with `!` prefix, for example:

- `!Go forward`
- `!Jump`
- `!Stop`
- `!Patrol`
- `!Attack`

### Teach New Commands

You can teach the bot new actions during gameplay:

```bash
!למד פקודה | Jump Twice | קפוץ פעמיים | bot.setControlState('jump', true); setTimeout(() => bot.setControlState('jump', false), 300);
```

---

## 🗃 Packaging & Sharing

To share this project:

- Zip the files (`SentinelAI_1_6.py`, `bot.js`, `learned_ai.json`, `README.md`)
- Upload to GitHub or share the ZIP file directly.

---

## 🌍 Notes

- The AI model runs locally – no cloud or external APIs.
- Everything happens in real-time.
- Fully open-source and customizable.

---

## ❤️ Created by Raz & ChatGPT

Feel free to contribute, improve, and share.
