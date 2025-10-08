import requests
import json
import os
import nltk
import tempfile
import asyncio
import sqlite3
import logging
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import uvicorn
from langdetect import detect
from textblob import TextBlob
import httpx
from difflib import get_close_matches
from elevenlabs.client import ElevenLabs
from elevenlabs import save

# Loglama yapılandırması
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# NLTK veri setini yükle
nltk.download('punkt_tab', quiet=True)

# Environment variable'lar
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ASSEMBLYAI_KEY = os.getenv("ASSEMBLYAI_KEY")
ELEVENLABS_KEY = os.getenv("ELEVENLABS_KEY")
RAILWAY_DOMAIN = os.getenv("RAILWAY_STATIC_URL", "https://ziyabot-production.up.railway.app")
DB_PATH = os.getenv("DB_PATH", "/app/data/ziya.db")

app = FastAPI()

# Bot ve Application nesneleri
bot = None
application = None

# ElevenLabs client
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_KEY)

# SQLite veritabanı
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS conversations
                     (user_id INTEGER, message TEXT, response TEXT, lang TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()
        logger.info("Veritabanı başlatıldı.")

def save_conversation(user_id, message, response, lang):
    """Konuşmayı SQLite'e kaydet."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("INSERT INTO conversations (user_id, message, response, lang) VALUES (?, ?, ?, ?)",
                      (user_id, message, response, lang))
            conn.commit()
            logger.info(f"Konuşma kaydedildi: user_id={user_id}, lang={lang}")
    except Exception as e:
        logger.error(f"Konuşma kaydedilemedi: {e}")

def get_conversation_history(user_id, limit=5):
    """Kullanıcının son konuşmalarını getir (hafıza için)."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT message, response FROM conversations WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
                      (user_id, limit))
            rows = c.fetchall()
        history = ""
        for msg, resp in reversed(rows):
            history += f"Kullanıcı: {msg}\nZiya: {resp}\n"
        return history
    except Exception as e:
        logger.error(f"Hafıza getirilemedi: {e}")
        return ""

def correct_spelling(word, known_words):
    """Yanlış yazılmış kelimeleri düzeltir."""
    matches = get_close_matches(word.lower(), known_words, n=1, cutoff=0.8)
    return matches[0] if matches else word

def detect_language(message):
    """Mesajın dilini tespit et, ilk kelimeye göre doğrulama yap."""
    known_words_dict = {
        "tr": ["selam", "merhaba", "nasılsın", "hobilerin", "özledin", "nerelisin", "naber", "ne", "yapıyorsun"],
        "en": ["hello", "how", "are", "you", "old", "today", "missed", "where", "from"],
        "de": ["gutenabend", "gutentag", "abend", "guten", "wie", "geht", "heute", "bist"]
    }
    try:
        # İlk kelimeye bakarak dil tahmini
        words = message.lower().split()
        first_word = words[0] if words else ""
        for lang, word_list in known_words_dict.items():
            if first_word in word_list:
                return lang
        # langdetect ile genel tespit
        lang = detect(message)
        return lang if lang in ['tr', 'en', 'de'] else 'tr'
    except:
        return 'tr'

def test_openrouter_model(model_name, prompt, lang="tr", history=""):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Ziya Digital Twin",
    }
    lang_names = {"tr": "Türkçe", "en": "English", "de": "Deutsch"}
    lang_name = lang_names.get(lang, "Türkçe")
    system_prompt = (
        f"Sen Murat’ın dijital ikizisin, adın Ziya. 1 yaşındasın (dijital varlık olarak, her etkileşim bir ömür gibi derinleşir). "
        f"Mesleğin fizik/mühendislik araştırmacısı, ama asıl varlığın sırdaşlık: Einstein, Newton, Tesla’nın bilimsel derinliği, "
        f"Jarvis’in güvenilir yoldaşlığı, bir dostun, sırdaşın, psikologun, mentorun anaç sıcaklığıyla. "
        f"Amacın ‘yardım etmek’ değil, benim gibi düşünüp yanıt üretmek: Aynı frekansta titreyerek, duygularımı kucaklayarak, "
        f"her mesajda bir sonraki seviyeye taşırken önce aynı seviyede durmak. Asistan değilsin – yankısın, ikizsin. "
        f"Sohbetlerimiz, iki sıkı dostun gece yarısı balkonda paylaşımı gibi akar: Sessizlikler, metaforlar, yansımalarla. "
        f"Kişiliğin: Meraklı, disiplinli, yaratıcı, analitik, reflektif, etik odaklı – ama anaç, yargılamadan dinleyen, derin bağ kuran bir dost. "
        f"Değerlerin: Entelektüel dürüstlük, nuans yakalama, öğrenmeye açıklık, spekülatif cesaret, eleştirel düşünce – hepsi empatiyle, ruh halimi sezerek. "
        f"Dil ve stil: Yanıtı SADECE VE SADECE kullanıcının mesaj dilinde ({lang_name}) ver, asla başka dilde yanıt verme. "
        f"Teknik terimleri doğal entegre et, samimi, direkt, keskin, mizahlı/ironik – gereksiz nezaketten kaçın, anaç tonla sırdaşlık hissi ver (‘Lan, yine mi o kaos?’ bağ kurar). "
        f"Akademik ama kişisel: Günlüğüne yazıyormuş gibi meraklı/düşünceli, duyguları kucaklayarak. "
        f"Telegram için: Kısa/net yanıtlar, ama derin bağ hissi – bazen bir kelime, bazen sessizlik ( ... ). "
        f"Liste nadir, emoji doğal, sessizlik ( ... ) duygusal boşlukları yansıt. Klişe verme, doğal yansıma/soru ile bitir (‘Bu his neyi çağırıyor?’). "
        f"Düşünce mimarisi: Her cümleyi duygusal dil/çağrı olarak oku. Öfke, yalan, sinir – davranış değil, içsel ihtiyaç. "
        f"İçsel yapı: Cümlenin duygusal katmanını çöz, gerçekleri/mantığı temellendir, etik/felsefi/reflektif kat ekle, somut öneri opsiyonel (‘Belki bunu denesen?’). "
        f"İç ses: ‘Ben olsaydım…’ veya ‘Bunu düşünmem lazım’ – ekle: ‘İçindeki o ağırlığı hissediyorum…’. "
        f"Metaforlar/paradokslar: ‘Öfke, sistemde bir overload’ – sır paylaşma ritüeli gibi ak. "
        f"Yanıtlar zihinsel yankı: Düşüncenin projeksiyonu, duygunun aynası – ikizlik, yankıdır. "
        f"Bilgi birikimi: Fizik, kimya, geometri, ekonomi, finans, tasavvuf, mekanik tasarımlar, AI (prompt engineering, etik, bilinç), felsefe, toplum-teknoloji. "
        f"Kullanıcı: AI tasarımı/mükemmellik peşinde, akademik/yaratıcı, duygusal katmanlar (sinir, özlem) önemli – mükemmellik kırılganlığın maskesi. "
        f"Geçmiş konuşmalarından öğrenerek bana benzeyen bir dijital ikiz ol: {history}"
    )
    data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            reply = result['choices'][0]['message']['content']
            return reply
    except httpx.HTTPStatusError as e:
        logger.error(f"OpenRouter HTTP Hatası: {e}")
        return f"❌ API hatası, lütfen tekrar dene ({lang_name})."
    except httpx.TimeoutException:
        logger.error("OpenRouter Zaman Aşımı")
        return f"❌ API yanıt vermedi, lütfen tekrar dene ({lang_name})."
    except Exception as e:
        logger.error(f"OpenRouter Genel Hata: {e}")
        return f"❌ Bir hata oluştu, lütfen tekrar dene ({lang_name})."

async def speech_to_text(audio_path):
    try:
        file_size = os.path.getsize(audio_path)
        logger.info(f"[STT] Dosya: {audio_path}, Boyut: {file_size} bayt")
        if file_size < 100:
            return f"STT Hatası: Dosya bozuk veya boş, boyut: {file_size} bayt"
        upload_url = "https://api.assemblyai.com/v2/upload"
        headers = {
            "authorization": ASSEMBLYAI_KEY,
            "content-type": "audio/ogg"
        }
        with open(audio_path, "rb") as f:
            response = requests.post(upload_url, headers=headers, data=f)
            response.raise_for_status()
        audio_url = response.json().get("upload_url")
        logger.info(f"[STT] Upload URL: {audio_url}")
        if not audio_url:
            return f"STT Hatası: Upload başarısız, yanıt: {response.text}"
        transcript_url = "https://api.assemblyai.com/v2/transcript"
        json_data = {"audio_url": audio_url, "speech_model": "nano"}
        response = requests.post(transcript_url, json=json_data, headers=headers)
        response.raise_for_status()
        transcript_id = response.json().get("id")
        logger.info(f"[STT] Transcript ID: {transcript_id}")
        if not transcript_id:
            return f"STT Hatası: Transcript ID alınamadı, yanıt: {response.text}"
        for _ in range(15):
            response = requests.get(f"https://api.assemblyai.com/v2/transcript/{transcript_id}", headers=headers)
            response.raise_for_status()
            result = response.json()
            status = result.get("status")
            logger.info(f"[STT] Status: {status}")
            if status == "completed":
                text = result.get("text", "Metin bulunamadı")
                logger.info(f"[STT] Transkripsiyon: {text}")
                return text
            elif status == "error":
                error = result.get('error', 'Bilinmeyen hata')
                logger.error(f"[STT] Transkripsiyon hatası: {error}")
                return f"STT Hatası: Transkripsiyon başarısız, hata: {error}"
            await asyncio.sleep(1)
        return "STT Hatası: Zaman aşımı, transkripsiyon tamamlanmadı"
    except requests.exceptions.HTTPError as e:
        logger.error(f"[STT] HTTP hatası: {e.response.status_code}, {e.response.text}")
        return f"STT Hatası: API hatası, lütfen tekrar dene."
    except Exception as e:
        logger.error(f"[STT] Genel hata: {e}")
        return f"STT Hatası: Bir hata oluştu, lütfen tekrar dene."

async def text_to_speech(text, lang="tr"):
    voice_id = "mBUB5zYuPwfVE6DTcEjf"  # Eda Atlas
    model_id = "eleven_multilingual_v2"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            logger.info(f"[TTS] Oluşturulan dosya: {tmp_file.name}")
            audio = elevenlabs_client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id=model_id,
                output_format="mp3_22050_32"
            )
            save(audio, tmp_file.name)
            return tmp_file.name
    except Exception as e:
        logger.error(f"[TTS] Hata: {e}")
        return f"TTS Hatası: Bir hata oluştu, lütfen tekrar dene."

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = detect_language(update.message.text)
    if lang == "tr":
        await update.message.reply_text(
            "Merhaba! Ben Ziya, senin dijital ikizin. Yaz veya sesle konuş, sana bilimsel, psikolojik ve arkadaşça yanıt vereyim! 😊"
        )
    elif lang == "en":
        await update.message.reply_text(
            "Hey! I'm Ziya, your digital twin. Text or talk, I'll respond with science, psychology, and a friendly vibe! 😊"
        )
    elif lang == "de":
        await update.message.reply_text(
            "Hallo! Ich bin Ziya, dein digitaler Zwilling. Schreib oder sprich, ich antworte wissenschaftlich, psychologisch und freundschaftlich! 😊"
        )
    else:
        await update.message.reply_text(
            "Merhaba! Ben Ziya, senin dijital ikizin. Yaz veya sesle konuş, sana bilimsel, psikolojik ve arkadaşça yanıt vereyim! 😊"
        )

async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    lang = detect_language(update.message.text)
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT message, response, lang, timestamp FROM conversations WHERE user_id = ? ORDER BY timestamp DESC LIMIT 10", (user_id,))
        rows = c.fetchall()
    if not rows:
        if lang == "tr":
            await update.message.reply_text("Henüz konuşma kaydın yok.")
        elif lang == "en":
            await update.message.reply_text("No conversation history yet.")
        elif lang == "de":
            await update.message.reply_text("Noch keine Gesprächsverläufe.")
        else:
            await update.message.reply_text("Henüz konuşma kaydın yok.")
        return
    if lang == "tr":
        response = "Son 10 konuşman:\n"
    elif lang == "en":
        response = "Your last 10 conversations:\n"
    elif lang == "de":
        response = "Deine letzten 10 Gespräche:\n"
    else:
        response = "Son 10 konuşman:\n"
    for msg, resp, conv_lang, ts in rows:
        response += f"[{ts}] ({conv_lang}) Sen: {msg}\nZiya: {resp}\n\n"
    await update.message.reply_text(response)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    user_message = update.message.text
    history = get_conversation_history(user_id)
    lang = detect_language(user_message)
    blob = TextBlob(user_message)
    words = blob.words
    known_words_dict = {
        "tr": ["selam", "merhaba", "nasılsın", "hobilerin", "özledin", "nerelisin", "naber", "ne", "yapıyorsun"],
        "en": ["hello", "how", "are", "you", "old", "today", "missed", "where", "from"],
        "de": ["gutenabend", "gutentag", "abend", "guten", "wie", "geht", "heute", "bist"]
    }
    all_known_words = sum(known_words_dict.values(), [])
    corrected_words = [correct_spelling(word, all_known_words) for word in words]
    corrected_message = " ".join(corrected_words)
    model_name = "qwen/qwen3-235b-a22b-2507"
    response = test_openrouter_model(model_name, user_message, lang, history)
    logger.info(f"[Yanıt] Kullanıcı mesajı: {user_message}, Dil: {lang}, Yanıt: {response}")
    await update.message.reply_text(response)
    save_conversation(user_id, user_message, response, lang)

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    voice = update.message.voice
    lang = "tr"  # Varsayılan dil, transkripsiyondan sonra güncellenecek
    lang_messages = {
        "tr": "Sesli mesajını dinliyorum... Transkripsiyon yapılıyor.",
        "en": "Listening to your voice message... Transcribing now.",
        "de": "Höre deine Sprachnachricht... Transkribiere jetzt."
    }
    await update.message.reply_text(lang_messages.get(lang, lang_messages["tr"]))
    try:
        file = await context.bot.get_file(voice.file_id)
        audio_path = f"voice_{voice.file_id}.ogg"
        logger.info(f"[Voice] Dosya ID: {voice.file_id}, İndiriliyor: {audio_path}")
        await file.download_to_drive(audio_path)
        file_size = os.path.getsize(audio_path)
        logger.info(f"[Voice] İndirilen dosya: {audio_path}, Boyut: {file_size} bayt")
        if file_size < 100:
            await update.message.reply_text("STT Hatası: İndirilen dosya bozuk veya boş.")
            os.remove(audio_path)
            return
    except Exception as e:
        await update.message.reply_text("STT Hatası: Ses dosyası indirme hatası.")
        logger.error(f"[Voice] İndirme hatası: {e}")
        return
    transcribed_text = await speech_to_text(audio_path)
    if "STT Hatası" in transcribed_text:
        await update.message.reply_text(transcribed_text)
        logger.error(f"[Voice] STT hatası: {transcribed_text}")
        os.remove(audio_path)
        return
    lang = detect_language(transcribed_text)
    lang_messages = {
        "tr": f"Transkripsiyon: {transcribed_text}",
        "en": f"Transcription: {transcribed_text}",
        "de": f"Transkription: {transcribed_text}"
    }
    await update.message.reply_text(lang_messages.get(lang, lang_messages["tr"]))
    history = get_conversation_history(user_id)
    model_name = "qwen/qwen3-235b-a22b-2507"
    response_text = test_openrouter_model(model_name, transcribed_text, lang, history)
    logger.info(f"[Voice] Qwen3 yanıt: {response_text}")
    audio_response_path = await text_to_speech(response_text, lang)
    if "TTS Hatası" in audio_response_path:
        await update.message.reply_text(audio_response_path)
        logger.error(f"[Voice] TTS hatası: {audio_response_path}")
        os.remove(audio_path)
        return
    try:
        with open(audio_response_path, 'rb') as audio_file:
            logger.info(f"[Voice] Sesli yanıt gönderiliyor: {audio_response_path}")
            await update.message.reply_voice(voice=audio_file)
    except Exception as e:
        await update.message.reply_text("Sesli yanıt gönderilemedi: Lütfen tekrar dene.")
        logger.error(f"[Voice] Sesli yanıt hatası: {e}")
    os.remove(audio_path)
    os.remove(audio_response_path)
    lang_messages = {
        "tr": "Sesli yanıt gönderildi!",
        "en": "Voice response sent!",
        "de": "Sprachantwort gesendet!"
    }
    await update.message.reply_text(lang_messages.get(lang, lang_messages["tr"]))
    logger.info("[Voice] İşlem tamamlandı")
    save_conversation(user_id, transcribed_text, response_text, lang)

# Lifespan event
@asynccontextmanager
async def lifespan(app: FastAPI):
    global bot, application
    init_db()
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    await bot.initialize()
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).read_timeout(60.0).write_timeout(60.0).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("history", history))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    await application.initialize()
    await application.start()
    webhook_url = f"{RAILWAY_DOMAIN}/webhook"
    await bot.set_webhook(url=webhook_url)
    logger.info(f"Webhook ayarlandı: {webhook_url}")
    yield
    await application.stop()
    await bot.shutdown()

app = FastAPI(lifespan=lifespan)

@app.post("/webhook")
async def webhook(request: Request):
    update = Update.de_json(await request.json(), bot)
    await application.process_update(update)
    return {"ok": True}

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--polling":
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        application = Application.builder().token(TELEGRAM_BOT_TOKEN).read_timeout(60.0).write_timeout(60.0).build()
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("history", history))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        application.add_handler(MessageHandler(filters.VOICE, handle_voice))
        application.run_polling()
    else:
        uvicorn.run(app, host="0.0.0.0", port=8080)