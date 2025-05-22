import telebot
import sqlite3
import threading


BOT_TOKEN = "Tokenbot"
bot = telebot.TeleBot(BOT_TOKEN)


def send_warning():
    chat_id = "chatid"

    conn = sqlite3.connect('violations.db')
    cursor = conn.cursor()

    # Ambil data terakhir
    cursor.execute("SELECT date, timestamp, image_path FROM violations ORDER BY id DESC LIMIT 1")
    result = cursor.fetchone()
    conn.close()

    if result:
        date, time, image_path = result

        caption = (
            f"<b>‼ WARNING THERE IS A SMOKING VIOLATION ‼</b>\n"
            "\n"
            f"Date 📆: {date}\n"
            f"Time ⏱: {time}\n"
        )

        try:
            with open(image_path, 'rb') as img:
                bot.send_photo(chat_id, photo=img, caption=caption, parse_mode="HTML")
        except FileNotFoundError:
            bot.send_message(chat_id, f"⚠️ File tidak ditemukan: {image_path}")
    else:
        bot.send_message(chat_id, "❌ Tidak ada data pelanggaran.")


def start_polling():
    bot.polling()


def start_bot_in_thread():
    bot_thread = threading.Thread(target=start_polling)
    bot_thread.daemon = True  # Pastikan bot thread berhenti saat program utama berhenti
    bot_thread.start()
