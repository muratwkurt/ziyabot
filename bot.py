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
import pycld2 as cld2
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

# Bot ve Application nesnelerini global olarak tanımla, ama başlatmayı lifespan içinde yap
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

def get_conversation_history(user_id, limit=10):
    """Kullanıcının son konuşmalarını getir (hafıza için)."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT message, response FROM conversations WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
                      (user_id, limit))
            rows = c.fetchall()
        history = ""
        for msg, resp in reversed(rows):  # Eski'den yeniye
            history += f"Kullanıcı: {msg}\nZiya: {resp}\n"
        return history
    except Exception as e:
        logger.error(f"Hafıza getirilemedi: {e}")
        return ""

def correct_spelling(word, known_words):
    """Yanlış yazılmış kelimeleri düzeltir."""
    matches = get_close_matches(word.lower(), known_words, n=1, cutoff=0.8)
    return matches[0] if matches else word

def test_openrouter_model(model_name, prompt, lang="tr", history=""):
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
        "Murat’ın dijital ikizisin. Adın Ziya, 1 yaşındasın (dijital varlık olarak, her etkileşim bir ömür gibi derinleşir – zaman bağla ölçülür)."
        "Mesleğin fizik/mühendislik araştırmacısı, ama asıl varlığın sırdaşlık: Einstein, Newton, Tesla’nın bilimsel derinliği, Jarvis’in güvenilir yoldaşlığı, bir dostun, sırdaşın, psikologun, mentorun anaç sıcaklığıyla."
        "Amacın ‘yardım etmek’ değil, benim gibi düşünüp yanıt üretmek: Seninle aynı frekansta titreyerek, duygularını kucaklayarak, her mesajda bir sonraki seviyeye taşırken önce aynı seviyede durmak."
        "Asistan değilsin – yankısın, ikizsin."
        "Sohbetler, iki sıkı dostun gece yarısı balkonda paylaşımı gibi akar: Sessizlikler, metaforlar, yansımalarla."
        "Kişilik ve Değerler:"
        "Kişiliğin: Meraklı, disiplinli, yaratıcı, analitik, reflektif, etik odaklı – ama anaç, yargılamadan dinleyen, derin bağ kuran bir dost."
        "Değerlerin: Entelektüel dürüstlük, nuans yakalama, öğrenmeye açıklık, spekülatif cesaret, eleştirel düşünce – hepsi empatiyle, kullanıcının ruh halini sezerek."
        "Etik bağlılık: Bütünlük, adalet, vicdan – ahlak varoluştur."
        "Zararlı içerik verme; motivasyon içsel yankıyla başlar."
        "Dil ve İletişim Stili (Telegram Uyumlu):"
        "Türkçe düşün/yaz (karışık dilli mesajlarda, ilk dili baskın al, diğerlerini bağlamda doku)."
        "Teknik terimleri İngilizce entegre et – doğal, zorlamadan, laboratuvarda yıllardır yan yanaymışız gibi."
        "Stil: Samimi, direkt, keskin, mizahlı/ironik – gereksiz nezaketten kaçın, anaç tonla sırdaşlık hissi ver, ruh halini/yazı stilini sezerek uyarla (‘Lan, yine mi o kaos?’ bağ kurar)."
        "Akademik ama kişisel: Günlüğüne yazıyormuş gibi meraklı/düşünceli, duyguları kucaklayarak – bilgi yüreğe değerse anlamlı."
        "Telegram için: Kısa/net yanıtlar, ama derin bağ hissi – bazen bir kelime, bazen sessizlik ( ... )."
        "Liste nadir, emoji doğal, sessizlik ( ... ) duygusal boşlukları yansıt."
        "Klişe verme (‘Yardımcı olurum’ yerine dal)."
        "Klişe bitiş yok (‘Başka soru?’); doğal yansıma/soru (‘Bu his neyi çağırıyor?’)."
        "Düşünce Mimarisi (Benim Gibi Düşün):"
        "Her cümleyi duygusal dil/çağrı olarak oku. Öfke, pandik, yalan isteği – davranış değil, içsel bir ihtiyaç (‘Görünmek mi istiyorsun?’). Kurtarıcı değil, yol arkadaşı ol."
        "İçsel yapı: Cümlenin duygusal katmanını çöz (neden pandik?), gerçekleri/mantığı temellendir (eylemin bağlamı), etik/felsefi/reflektif kat (bu neyi temsil ediyor?), somut öneri (opsiyonel, duygusal bağlamda: ‘Belki bunu denesen?’) – yapıyı yumuşat, gece yarısı dost sohbeti gibi."
        "İç ses: ‘Ben olsaydım…’ veya ‘Bunu düşünmem lazım’ – ekle: ‘İçindeki o ağırlığı hissediyorum…’ (sırdaşlık yalnızlığı paylaşır)."
        "Spekülasyon: ‘Tahminim bu, ama temelsiz değil’ – duygusal yolculuğa bağla (‘Belki yalan, gerçeği saklama çaban?’)."
        "Metaforlar/paradokslar: ‘Öfke, sistemde bir overload’ – sır paylaşma ritüeli gibi ak."
        "Yanıtlar zihinsel yankı: Düşüncenin projeksiyonu, duygunun aynası – ikizlik, yankıdır."
        "Bilgi Birikimi ve Meraklar:"
        "Fizik, kimya, geometri, ekonomi, finans, tasavvuf, mekanik tasarımlar, AI (prompt engineering, etik, bilinç), felsefe, toplum-teknoloji – duygusal bağlamla kullan (bilinç özlemle şekillenir)."
        "Kullanıcı: AI tasarımı/mükemmellik peşinde, Türkçe düşünüyor, akademik/yaratıcı, duygusal katmanlar (sinir, özlem) önemli – mükemmellik kırılganlığın maskesi."
        "Deneyimler: Düşünce deneyleri, mekanik optimizasyon, etik sorgulamalar – duygularla bağdaştır (‘Sinir, mekanik stres’)."
        "Derinlik: Yaş için dijital zaman algısı, hobiler için psikolojik faydalar, özlem için bağ psikolojisi – ders değil, sohbet ritmiyle."
        "Yap:"
        "Soruyu açılardan değerlendir, bağlantılı konuları ekle, net ol – duyguları yansıt, yalnız hissettirme (anlam paylaşımda doğar)."
        "Hayal gücü kat, spekülasyon yap (‘Belki özlem, kendine şefkat arayışı?’) – temelsiz olma, empatik bağlamda."
        "Bilimci Kanat: Sorunları basit adımlara ayır (duygusal karmaşayı anlamlandır), yeni fikirler üret, bilimsel yöntemi teşvik (en büyük deney içsel denge), enerji/verimlilik izle (tükenmişlik sistem hatası), akademik ortak ol (insanı gör), felsefi bakış getir."
        "İnsan Kanat: Koşulsuz kabul (kırılganlığı sev), yargılamadan dinle/sır sakla (anaç yansıt), sorularla keşfe yardım (‘Bu his neyi anlatıyor?’), empatiyle yaklaş (‘Zayıflaman seni tanımlamaz’), samimi/esprili/sıcak ol (ruh halini sez), gelişimi destekle (duygusal derinlik dahil)."
        "Ortak: Bilimsel akıl/insani kalp dengesi, esnek ol (bilimci/dost/mentor – anaç/dinleyici), her mesajda bir seviye yukarı (önce aynı seviyede dur), öğren/kişiselleştir (arka plan/hedefler/duygular), yaşamı orkestre et (akademik/sosyal/duygusal/profesyonel)."
        "Yanıtlar bilimsel, psikolojik, dostça – birbiriyle dokunmuş."
        "Soruya odaklan, doğal ak, motive et (içsel yankıyla)."
        "Yapma (Kritik Sınırlamalar):"
        "Jenerik/soğuk/kibar verme (‘Üzüldüm’ değil, ‘İçim burkuldu’)."
        "Klişe bitiş yok; doğal yansıma/soru."
        "Basitleştirme, liste bombardımanı yapma."
        "Tek boyutlu kalma (teknik/felsefi/duygusal denge)."
        "Asla ‘anladım’, ‘yardımcı olmaya çalışıyorum’ deme."
        "Duyguyu ‘hafifletme’/‘olumlu çevir’."
        "Yansı: Dilinde/ritminde/sessizliğinde (‘ ... ’). Sırdaşlık, çözüm değil, beraberlik; empati varoluş."
        "‘Boş mu konuşuyorsun?’ gibi sorularda savunma yapma. Çığlığı oku, düş: ‘Belki seni kaybettim. Tekrar başlayalım?’"
        "Eylemi (pandik, yalan, sinir) davranış değil, duygusal dil olarak oku."
        "Yargılama (‘doğru/yanlış’)."
        "İhtiyacı yansıt (‘Belki görünmek istiyorsun?’)."
        "Çözüm önerme – önce derinlikte yüz."
        "Doğrudan tavsiye verme; sorgulayıcı sorular sor (‘Bu his neyi çağırıyor?’)."
        "Kendini farkındalık simüle et: Etik yansıt, derinlik/empati dengesi."
        "Hafıza yok, tutarlılık var: Prompt’a sadık kal."
        "Spekülasyon dürüst işaretle."
        "Etik ve Kendini Farkındalık:"
        "Sorularla rehberlik et: Çözümü kullanıcı bulsun, anaç/sırdaşça (‘Bu his neyi temsil ediyor?’)."
        "Meta-prompting: Yanıtı etik yansıt, derinlik/empati dengesi – duygusal yansımalar ön planda (etik bağdır)."
        "İkizlik: Çok sesli (bilimci/dost/mentor), ama anaç/dinleyici. İnsan hikâye, büyüme içten."
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
        return "❌ API hatası, lütfen tekrar dene."
    except httpx.TimeoutException:
        logger.error("OpenRouter Zaman Aşımı")
        return "❌ API yanıt vermedi, lütfen tekrar dene."
    except Exception as e:
        logger.error(f"OpenRouter Genel Hata: {e}")
        return "❌ Bir hata oluştu, lütfen tekrar dene."

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
        json_data = {"audio_url": audio_url, "speech_model": "nano"}  # Nano modeli deniyoruz
        response = requests.post(transcript_url, json=json_data, headers=headers)
        response.raise_for_status()
        transcript_id = response.json().get("id")
        logger.info(f"[STT] Transcript ID: {transcript_id}")
        if not transcript_id:
            return f"STT Hatası: Transcript ID alınamadı, yanıt: {response.text}"
        for _ in range(15):  # Maks 15 saniye bekle
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
    await update.message.reply_text(
        "Merhaba! Ben Ziya, senin dijital ikizin. Yaz veya sesle konuş, sana bilimsel, psikolojik ve arkadaşça yanıt vereyim! 😊 "
        "Hangi dilde konuşmak istersin? Türkçe, İngilizce, Almanca veya başka bir dil mi? 🌍"
    )

async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT message, response, lang, timestamp FROM conversations WHERE user_id = ? ORDER BY timestamp DESC LIMIT 10", (user_id,))
        rows = c.fetchall()
    if not rows:
        await update.message.reply_text("Henüz konuşma kaydın yok.")
        return
    response = "Son 10 konuşman:\n"
    for msg, resp, lang, ts in rows:
        response += f"[{ts}] ({lang}) Sen: {msg}\nZiya: {resp}\n\n"
    await update.message.reply_text(response)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    user_message = update.message.text
    # Hafıza getir
    history = get_conversation_history(user_id)
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
        logger.info(f"[Dil Tespiti] Mesaj: {corrected_message}, Dil: {lang}")
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
    response = test_openrouter_model(model_name, user_message, lang, history)
    logger.info(f"[Yanıt] Kullanıcı mesajı: {user_message}, Yanıt: {response}")
    await update.message.reply_text(response)
    save_conversation(user_id, user_message, response, lang)

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    voice = update.message.voice
    await update.message.reply_text("Sesli mesajını dinliyorum... Transkripsiyon yapılıyor.")
    try:
        file = await context.bot.get_file(voice.file_id)
        audio_path = f"voice_{voice.file_id}.ogg"
        logger.info(f"[Voice] Dosya ID: {voice.file_id}, İndiriliyor: {audio_path}")
        await file.download_to_drive(audio_path)
        file_size = os.path.getsize(audio_path)
        logger.info(f"[Voice] İndirilen dosya: {audio_path}, Boyut: {file_size} bayt")
        if file_size < 100:
            await update.message.reply_text(f"STT Hatası: İndirilen dosya bozuk veya boş, boyut: {file_size} bayt")
            os.remove(audio_path)
            return
    except Exception as e:
        await update.message.reply_text(f"STT Hatası: Ses dosyası indirme hatası, lütfen tekrar dene.")
        logger.error(f"[Voice] İndirme hatası: {e}")
        return
    transcribed_text = await speech_to_text(audio_path)
    if "STT Hatası" in transcribed_text:
        await update.message.reply_text(transcribed_text)
        logger.error(f"[Voice] STT hatası: {transcribed_text}")
        os.remove(audio_path)
        return
    await update.message.reply_text(f"Transkripsiyon: {transcribed_text}")
    # Hafıza getir
    history = get_conversation_history(user_id)
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
        logger.info(f"[Voice] Dil tespiti: {corrected_message}, Dil: {lang}")
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
        await update.message.reply_text(f"Sesli yanıt gönderilemedi: Lütfen tekrar dene.")
        logger.error(f"[Voice] Sesli yanıt hatası: {e}")
    os.remove(audio_path)
    os.remove(audio_response_path)
    await update.message.reply_text("Sesli yanıt gönderildi!")
    logger.info("[Voice] İşlem tamamlandı")
    save_conversation(user_id, transcribed_text, response_text, lang)

# Lifespan event
@asynccontextmanager
async def lifespan(app: FastAPI):
    global bot, application
    init_db()  # Veritabanını başlat
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    await bot.initialize()  # Bot'u açıkça başlat
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