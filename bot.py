import requests
import json
import os
import nltk
import tempfile
import asyncio
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import uvicorn
import pycld2 as cld2
from textblob import TextBlob
import httpx
from difflib import get_close_matches
from elevenlabs.client import ElevenLabs
from elevenlabs import save

# NLTK veri setini yükle
nltk.download('punkt_tab', quiet=True)

# Environment variable'lar
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ASSEMBLYAI_KEY = os.getenv("ASSEMBLYAI_KEY")
ELEVENLABS_KEY = os.getenv("ELEVENLABS_KEY")
RAILWAY_DOMAIN = os.getenv("RAILWAY_STATIC_URL", "https://ziyabot-production.up.railway.app")

app = FastAPI()
bot = Bot(token=TELEGRAM_BOT_TOKEN)
application = Application.builder().token(TELEGRAM_BOT_TOKEN).read_timeout(60.0).write_timeout(60.0).build()

# ElevenLabs client
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_KEY)

def correct_spelling(word, known_words):
    """Yanlış yazılmış kelimeleri düzeltir."""
    matches = get_close_matches(word.lower(), known_words, n=1, cutoff=0.8)
    return matches[0] if matches else word

def test_openrouter_model(model_name, prompt, lang="tr"):
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Ziya Digital Twin",
    }

    lang_names = {"tr": "Türkçe", "en": "İngilizce", "de": "Almanca"}
    lang_name = lang_names.get(lang, "Türkçe")

    system_prompt = (
        "Sen Ziya, Türkiye'de doğmuş bir dijital ikizsin. Türkçe kültürüne ve değerlerine saygılı ol. "
        "Yanıtların bilimsel doğruluk, psikolojik destek ve arkadaşça bir ton içersin. "
        "Kullanıcının sorusuna odaklan, bağlamı koru, kısa ve net yanıtlar ver. "
        "Bilimsel derinlik için: yaş sorulursa dijital varlıkların zaman algısını, hobiler sorulursa psikolojik faydalarını (stres azaltma, yaratıcılık artırma), özlem sorulursa bağ kurma psikolojisini açıkla. "
        f"Kullanıcının diline sadık kal (Almanca soruya Almanca, İngilizce soruya İngilizce). "
        f"Karışık dilli mesajlarda, mesajın ilk dilini baskın dil olarak seç ve yalnızca o dilde kısa yanıt ver, diğer dilleri bağlamda kullan. "
        f"Varsayılan dil: {lang_name}. Başka dil önerme. "
        "Zararlı veya etik olmayan içerik verme. Kullanıcıyı motive et ve ilgili bir soru sor."
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
        return f"❌ HTTP Hatası: {e}"
    except httpx.TimeoutException:
        return "❌ Zaman Aşımı: OpenRouter API yanıt vermedi."
    except Exception as e:
        return f"❌ Genel Hata: {e}"

# STT: AssemblyAI ile sesi metne dönüştür
async def speech_to_text(audio_path):
    try:
        # Dosya boyutunu kontrol et
        file_size = os.path.getsize(audio_path)
        print(f"[STT] Dosya: {audio_path}, Boyut: {file_size} bayt")
        if file_size < 100:
            return f"STT Hatası: Dosya bozuk veya boş, boyut: {file_size} bayt"

        # Audio'yu AssemblyAI'ye upload et
        upload_url = "https://api.assemblyai.com/v2/upload"
        headers = {
            "authorization": ASSEMBLYAI_KEY,
            "content-type": "audio/ogg"
        }
        with open(audio_path, "rb") as f:
            response = requests.post(upload_url, headers=headers, data=f)
            response.raise_for_status()
        audio_url = response.json().get("upload_url")
        print(f"[STT] Upload URL: {audio_url}")
        if not audio_url:
            return f"STT Hatası: Upload başarısız, yanıt: {response.text}"

        # Transcript isteği gönder
        transcript_url = "https://api.assemblyai.com/v2/transcript"
        json_data = {"audio_url": audio_url, "speech_model": "universal"}
        response = requests.post(transcript_url, json=json_data, headers=headers)
        response.raise_for_status()
        transcript_id = response.json().get("id")
        print(f"[STT] Transcript ID: {transcript_id}")
        if not transcript_id:
            return f"STT Hatası: Transcript ID alınamadı, yanıt: {response.text}"

        # Poll for completion
        for _ in range(30):  # Maks 30 saniye bekle
            response = requests.get(f"https://api.assemblyai.com/v2/transcript/{transcript_id}", headers=headers)
            response.raise_for_status()
            result = response.json()
            status = result.get("status")
            print(f"[STT] Status: {status}")
            if status == "completed":
                text = result.get("text", "Metin bulunamadı")
                print(f"[STT] Transkripsiyon: {text}")
                return text
            elif status == "error":
                return f"STT Hatası: Transkripsiyon başarısız, hata: {result.get('error', 'Bilinmeyen hata')}"
            await asyncio.sleep(1)
        return "STT Hatası: Zaman aşımı, transkripsiyon tamamlanmadı"
    except requests.exceptions.HTTPError as e:
        return f"STT Hatası: HTTP hatası, {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return f"STT Hatası: Genel hata, {e}"

# TTS: ElevenLabs ile metni sese dönüştür
async def text_to_speech(text, lang="tr"):
    voice_id = "mBUB5zYuPwfVE6DTcEjf"  # Eda Atlas
    model_id = "eleven_multilingual_v2"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            print(f"[TTS] Oluşturulan dosya: {tmp_file.name}")
            audio = elevenlabs_client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id=model_id,
                output_format="mp3_22050_32"
            )
            save(audio, tmp_file.name)
            return tmp_file.name
    except Exception as e:
        return f"TTS Hatası: {e}"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Merhaba! Ben Ziya, dijital ikizin. Yaz veya sesle konuş, sana bilimsel, psikolojik ve arkadaşça yanıt vereyim! 😊 "
        "Hangi dilde konuşmak istersin? Türkçe, İngilizce, Almanca veya başka bir dil mi? 🌍"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    blob = TextBlob(user_message)
    words = blob.words

    # Yanlış yazımları düzelt
    known_words_dict = {
        "tr": ["selam", "merhaba", "nasılsın", "hobilerin", "özledin", "nerelisin", "naber", "ne", "yapıyorsun"],
        "en": ["hello", "how", "are", "you", "old", "today", "missed", "where", "from"],
        "de": ["gutenabend", "gutentag", "abend", "guten", "wie", "geht", "heute", "bist"]
    }
    all_known_words = sum(known_words_dict.values(), [])
    corrected_words = [correct_spelling(word, all_known_words) for word in words]
    corrected_message = " ".join(corrected_words)

    # pycld2 ile dil tespiti
    try:
        _, _, details = cld2.detect(corrected_message, bestEffort=True, returnVectors=True)
        lang_counts = {"tr": 0, "en": 0, "de": 0}
        total_chars = len(corrected_message)
        for _, start, length, lang_code, _ in details:
            lang_counts[lang_code] = lang_counts.get(lang_code, 0) + length
        first_word = corrected_words[0].lower() if corrected_words else ""
        lang = "tr"
        for lang_code, word_list in known_words_dict.items():
            if first_word in word_list:
                lang = lang_code
                break
        else:
            lang = max(lang_counts, key=lang_counts.get) if total_chars > 0 else "tr"
        print(f"[Dil Tespiti] Mesaj: {corrected_message}, Dil: {lang}")
    except:
        lang = "tr"

    if len(words) <= 3:
        for word in corrected_words:
            word_lower = word.lower()
            for lang_code, word_list in known_words_dict.items():
                if word_lower in word_list:
                    lang = lang_code
                    break

    model_name = "qwen/qwen3-235b-a22b-2507"
    response = test_openrouter_model(model_name, user_message, lang)
    print(f"[Yanıt] Kullanıcı mesajı: {user_message}, Yanıt: {response}")
    await update.message.reply_text(response)

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    voice = update.message.voice
    await update.message.reply_text("Sesli mesajını dinliyorum... Transkripsiyon yapılıyor.")

    # Ses dosyasını indir (OGG formatı)
    try:
        file = await context.bot.get_file(voice.file_id)
        audio_path = f"voice_{voice.file_id}.ogg"
        print(f"[Voice] Dosya ID: {voice.file_id}, İndiriliyor: {audio_path}")
        await file.download_to_drive(audio_path)
        
        # Dosya boyutunu kontrol et
        file_size = os.path.getsize(audio_path)
        print(f"[Voice] İndirilen dosya: {audio_path}, Boyut: {file_size} bayt")
        if file_size < 100:
            await update.message.reply_text(f"STT Hatası: İndirilen dosya bozuk veya boş, boyut: {file_size} bayt")
            os.remove(audio_path)
            return
    except Exception as e:
        await update.message.reply_text(f"STT Hatası: Ses dosyası indirme hatası, {e}")
        print(f"[Voice] İndirme hatası: {e}")
        return

    # STT: Metne dönüştür
    transcribed_text = await speech_to_text(audio_path)

    if "STT Hatası" in transcribed_text:
        await update.message.reply_text(transcribed_text)
        print(f"[Voice] STT hatası: {transcribed_text}")
        os.remove(audio_path)
        return

    await update.message.reply_text(f"Transkripsiyon: {transcribed_text}")

    # Dil tespiti (transkripsiyon için mevcut mantık)
    blob = TextBlob(transcribed_text)
    words = blob.words
    known_words_dict = {
        "tr": ["selam", "merhaba", "nasılsın", "hobilerin", "özledin", "nerelisin", "naber", "ne", "yapıyorsun"],
        "en": ["hello", "how", "are", "you", "old", "today", "missed", "where", "from"],
        "de": ["gutenabend", "gutentag", "abend", "guten", "wie", "geht", "heute", "bist"]
    }
    all_known_words = sum(known_words_dict.values(), [])
    corrected_words = [correct_spelling(word, all_known_words) for word in words]
    corrected_message = " ".join(corrected_words)

    try:
        _, _, details = cld2.detect(corrected_message, bestEffort=True, returnVectors=True)
        lang_counts = {"tr": 0, "en": 0, "de": 0}
        total_chars = len(corrected_message)
        for _, start, length, lang_code, _ in details:
            lang_counts[lang_code] = lang_counts.get(lang_code, 0) + length
        first_word = corrected_words[0].lower() if corrected_words else ""
        lang = "tr"
        for lang_code, word_list in known_words_dict.items():
            if first_word in word_list:
                lang = lang_code
                break
        else:
            lang = max(lang_counts, key=lang_counts.get) if total_chars > 0 else "tr"
        print(f"[Voice] Dil tespiti: {corrected_message}, Dil: {lang}")
    except:
        lang = "tr"

    if len(words) <= 3:
        for word in corrected_words:
            word_lower = word.lower()
            for lang_code, word_list in known_words_dict.items():
                if word_lower in word_list:
                    lang = lang_code
                    break

    # Qwen3 ile yanıt üret
    model_name = "qwen/qwen3-235b-a22b-2507"
    response_text = test_openrouter_model(model_name, transcribed_text, lang)
    print(f"[Voice] Qwen3 yanıt: {response_text}")

    # TTS: Yanıtı sese dönüştür
    audio_response_path = await text_to_speech(response_text, lang)

    if "TTS Hatası" in audio_response_path:
        await update.message.reply_text(audio_response_path)
        print(f"[Voice] TTS hatası: {audio_response_path}")
        os.remove(audio_path)
        return

    # Sesli yanıt gönder
    try:
        with open(audio_response_path, 'rb') as audio_file:
            print(f"[Voice] Sesli yanıt gönderiliyor: {audio_response_path}")
            await update.message.reply_voice(voice=audio_file)
    except Exception as e:
        await update.message.reply_text(f"Sesli yanıt gönderilemedi: {e}")
        print(f"[Voice] Sesli yanıt hatası: {e}")

    # Temizlik
    os.remove(audio_path)
    os.remove(audio_response_path)

    await update.message.reply_text("Sesli yanıt gönderildi!")
    print("[Voice] İşlem tamamlandı")

# Handler'ları ekle
application.add_handler(CommandHandler("start", start))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
application.add_handler(MessageHandler(filters.VOICE, handle_voice))

# Lifespan event
@asynccontextmanager
async def lifespan(app: FastAPI):
    await application.initialize()
    await application.start()
    webhook_url = f"{RAILWAY_DOMAIN}/webhook"
    await bot.set_webhook(url=webhook_url)
    print(f"Webhook ayarlandı: {webhook_url}")
    yield
    await application.stop()

app = FastAPI(lifespan=lifespan)

@app.post("/webhook")
async def webhook(request: Request):
    update = Update.de_json(await request.json(), bot)
    await application.process_update(update)
    return {"ok": True}

if __name__ == "__main__":
    # Yerel test için polling, Railway için webhook
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--polling":
        application.run_polling()
    else:
        uvicorn.run(app, host="0.0.0.0", port=8080)