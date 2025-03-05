import telebot
import urllib.parse
import requests
import json

bot = telebot.TeleBot('') #the key for telebot

@bot.message_handler(commands=['start'])
def start_message(message):
    #bot.reply_to(message, 'hello')
    bot.send_message(message.chat.id, 'hello')

@bot.message_handler(func=lambda message:True)
def echo_all(message):
    #bot.reply_to(message, message.text)
    try:
        encode_text = urllib.parse.quote(message.text)
        response = requests.post('http://localhost:8000/chat?query='
            +encode_text, timeout=100)
        if response.status_code == 200:
            aisay = json.loads(response.text)
            if "msg" in aisay:
                bot.reply_to(message, aisay["msg"]["output"].encode('utf-8'))
                #audio_path = f"{aisay['id']}.mp3"
            else:
                bot.reply_to(message, "Sorry, I don't know how to reply you.")
    except requests.RequestException as e:
        bot.reply_to(message, "Sorry, I don't know how to reply you. 2")

bot.infinity_polling()