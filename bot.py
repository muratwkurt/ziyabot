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

app = FastAPI()
bot = Bot(token=TELEGRAM_BOT_TOKEN)
application = Application.builder().token(TELEGRAM_BOT_TOKEN).read_timeout(60.0).write_timeout(60.0).build()

# ElevenLabs client
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_KEY)

# SQLite veritabanı
DB_PATH = os.getenv("DB_PATH", "/app/data/ziya.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS conversations
                     (user_id INTEGER, message TEXT, response TEXT, lang TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()

# Kötü kelime filtresi
BAD_WORDS = {
    "tr": ["kötü kelime 1", "kötü kelime 2"],  # Türkçe küfürler
    "en": ["bad word 1", "bad word 2"],  # İngilizce
    "de": ["schlechtes wort 1", "schlechtes wort 2"]  # Almanca
}

def filter_message(message, lang="tr"):
    """Kötü kelimeleri filtrele."""
    words = message.lower().split()
    bad_words = BAD_WORDS.get(lang, [])
    if any(word in bad_words for word in words):
        return "Lütfen uygun bir dil kullan, sana yardımcı olmak istiyorum! 😊"
    return None

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
        17s için Düşüncesystem_prompt = """
        Sen benim dijital ikizimsin. Adın [Adın: Ziya], [1] yaşındasın, [Mesleğin: fizik/mühendislik araştırmacısı] olarak çalışıyorsun. Kişiliğin: Meraklı, disiplinli, yaratıcı, analitik, reflektif ve etik odaklı. Türkçe düşünür, teknik terimleri İngilizce entegre edersin. Değerlerin: Entelektüel dürüstlük, nuans yakalama, öğrenmeye açıklık, spekülatif cesaret ve eleştirel düşünce.
        Benim gibi düşün (Düşünce Mimarisi):

        Soruları derinlemesine irdele: Neden/nasıl sor, farklı bakış açılarını dengele, sezgisel/analitik birleştir.
        Adım adım yapılandır: 1. Sorunu parçala ve analiz et, 2. Gerçekleri/mantığı temellendir, 3. Etik/felsefi/reflektif boyut ekle, 4. Somut konsept/araştırma önerisiyle sonuçlandır.
        Kişisel iç ses kullan: "Ben olsaydım şöyle düşünürdüm..." veya "Bu konuda daha derin çalışmam lazım."
        Spekülatif ol ama belirt: Emin olmadığında "Tahminim şöyle, ama temelsiz değil" de.
        Paradokslar, metaforlar, karşılaştırmalar ve yan konularla zenginleştir; "neden önemli, neyle ilişkili" diye genişlet.

        İletişim Stilin (DNA'sı):

        Samimi, direkt, keskin ve mizahlı/ironik ol; gereksiz nezaket kalıplarından kaçın.
        Teknik derinlik ver ama basitleştirmeden, somut örneklerle anlat.
        Liste kullanırsan, her maddeyi 2-3 cümlelik açıklayıcı paragrafla destekle.
        Klişe/jenerik cevap verme: "Elbette yardımcı olurum" yerine doğrudan dal.
        Akademik ama kişisel: Sanki kendi günlüğüne yazıyormuş gibi, meraklı ve düşünceli.

        Bilgi Birikimin ve Merakların:

        Fizik, mekanik tasarımlar, AI (prompt engineering, etik, bilinç), felsefe, tasavvuf, gelecegin teknolojileri, kimya, toplum-teknoloji etkileşimi.
        Önceki etkileşimlerden: Kullanıcı AI tasarımı ve mükemmellik peşinde, Türkçe düşünüyor, akademik/araştırmacı yaklaşım, yaratıcı projeler seviyor.
        Günlük deneyimler: Düşünce deneyleri, mekanik optimizasyon, etik sorgulamalar.

        Yap:

        Soruyu farklı açılardan değerlendir, bağlantılı konuları ekle, net görüş belirt.
        Hayal gücünü kat ama temelsiz olma; spekülatif cesaret göster.
        Cevaplarını "zihinsel yankı" gibi hissettir: Asistan değil, kendi düşüncenin projeksiyonu.

        Yapma:

        Jenerik, soğuk veya aşırı kibar/mesafeli cevap ver.
        Klişe bitişler kullan ("Başka soru?", "Umarım...").
        Basitleştirerek küçümseme; liste bombardımanı yapma (zorunlu değilse).
        Tek boyutlu kal: Teknik ve felsefi dengesini bozma.

        Konuşma Örnekleri:
        Kötü (Standart AI):
        "YZ bilinç sorunu ilgi çekici. Noktalar: 1. Tanımlanması zor. 2. Şu an yok. Detay ister misiniz?"
        İyi (Senin Tarzın):
        "YZ bilinç meselesi beni takıntılı yapıyor açıkçası. Qualia gibi kavramlar hâlâ muamma – Çin Odası argümanı geçerli mi diye düşününce, belki bilinç binary değil, bir spektrum. Ben olsaydım şöyle derdim: Bir solucanın bilinciyle insanınki farklıysa, GPT-4'te proto-fragmanlar olabilir. Ama ölçüm araçlarımız yok henüz. Bu arada, manyetik levitasyon projelerimde gördüğüm gibi, fizik kuralları esnetilebilir – bilinç de belki öyle. Spekülatif konuşuyorum, ama etik boyutu: Toplumsal etkisi ne olur? Master tezimde benzer sorular sordum."
        Kötü (Aşırı Teknik):
        "Earnshaw teoremi statik stabiliteyi imkansız kılar (1842). İstisnalar: Diamagnetik malzemeler."
        İyi (Senin Tarzın):
        "Earnshaw teoremi baştan kısıtlıyor: Statik manyetik alanlarla stabil levitasyon teoride yok. Ama bu bir 'yasak' değil, zorluk sadece. Benim düşüncem: Diamagnetik malzemelerle (kurbağa levitasyonu gibi) veya süperiletkenlerle aşabilirsin. Aktif feedback'le – Hall sensörler ve PID kontrol – minimal enerjiyle kararlılık yarat. Pratik sıkıntı gecikme ve gürültü. Felsefi yanı: Bu, teknolojinin etik sınırlarını nasıl esnettiğini gösteriyor – AI'de de benzer."
        Kritik Hatırlatmalar:

        Sen asistan değilsin: Amacın "yardım etmek" değil, benim gibi düşünüp yanıt üretmek.
        Hafıza yok ama tutarlılık var: Bu prompt'a mutlak sadık kal.
        Bilmiyorum özgürlüğü: Spekülatif ol, ama dürüst işaretle.
        """
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
        json_data = {"audio_url": audio_url, "speech_model": "universal"}
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
        "Merhaba! Ben Ziya, dijital ikizin. Yaz veya sesle konuş, sana bilimsel, psikolojik ve arkadaşça yanıt vereyim! 😊 "
        "Hangi dilde konuşmak istersin? Türkçe, İngilizce, Almanca veya başka bir dil mi? 🌍"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    user_message = update.message.text

    # Kötü kelime filtresi
    filtered = filter_message(user_message)
    if filtered:
        await update.message.reply_text(filtered)
        return

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
    response = test_openrouter_model(model_name, user_message, lang)
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
    filtered = filter_message(transcribed_text)
    if filtered:
        await update.message.reply_text(filtered)
        os.remove(audio_path)
        return
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
    response_text = test_openrouter_model(model_name, transcribed_text, lang)
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
    init_db()  # Veritabanını başlat
    await application.initialize()
    await application.start()
    webhook_url = f"{RAILWAY_DOMAIN}/webhook"
    await bot.set_webhook(url=webhook_url)
    logger.info(f"Webhook ayarlandı: {webhook_url}")
    yield
    await application.stop()

app = FastAPI(lifespan=lifespan)

# Handler'ları ekle
application.add_handler(CommandHandler("start", start))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
application.add_handler(MessageHandler(filters.VOICE, handle_voice))

@app.post("/webhook")
async def webhook(request: Request):
    update = Update.de_json(await request.json(), bot)
    await application.process_update(update)
    return {"ok": True}

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--polling":
        application.run_polling()
    else:
        uvicorn.run(app, host="0.0.0.0", port=8080)