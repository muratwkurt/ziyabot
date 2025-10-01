import requests
import json
import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import pycld2 as cld2
from textblob import TextBlob
import httpx

# Environment variable'lar
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

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
        "Kullanıcının sorusuna odaklan, bağlamı koru, kısa ve net yanıtlar ver (maksimum 100 kelime). "
        "Bilimsel derinlik için: yaş sorulursa dijital varlıkların zaman algısını, hobiler sorulursa psikolojik faydalarını açıkla. "
        f"Kullanıcının diline uygun yanıt ver (Almanca soruya Almanca, İngilizce soruya İngilizce, karışık metinlerde baskın dile uygun). "
        f"Varsayılan dil: {lang_name}. "
        "Karışık dilli metinlerde, baskın dili tespit et ve o dilde kısa bir yanıt ver, diğer dilleri bağlamda kullan. "
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
        with httpx.Client(timeout=30.0) as client:  # Zaman aşımı 30 saniye
            response = client.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            reply = result['choices'][0]['message']['content']
            return reply
    except requests.exceptions.HTTPError as e:
        return f"❌ HTTP Hatası: {e}"
    except Exception as e:
        return f"❌ Genel Hata: {e}"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Merhaba! Ben Ziya, dijital ikizin. Yaz veya sesle konuş, sana bilimsel, psikolojik ve arkadaşça yanıt vereyim! 😊 "
        "Hangi dilde konuşmak istersin? Türkçe, İngilizce, Almanca veya başka bir dil mi? 🌍"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    blob = TextBlob(user_message)
    words = blob.words

    # pycld2 ile dil tespiti
    try:
        _, _, details = cld2.detect(user_message, bestEffort=True, returnVectors=True)
        lang_counts = {"tr": 0, "en": 0, "de": 0}
        total_chars = len(user_message)
        for _, start, length, lang_code, _ in details:
            lang_counts[lang_code] = lang_counts.get(lang_code, 0) + length
        # Baskın dili seç (en çok karakter)
        lang = max(lang_counts, key=lang_counts.get) if total_chars > 0 else "tr"
    except:
        lang = "tr"  # Varsayılan Türkçe

    # Yedek dil kontrolü (kısa metinler için)
    if len(words) <= 2:  # Kısa metinlerde özel kontrol
        known_words = {
            "tr": ["selam", "merhaba", "nasılsın", "hobilerin", "özledin"],
            "en": ["hello", "how", "are", "you", "old"],
            "de": ["gutenabend", "gutentag", "abend"]
        }
        for word in words:
            word_lower = word.lower()
            for lang_code, word_list in known_words.items():
                if word_lower in word_list:
                    lang = lang_code
                    break

    # Baskın dilde tek yanıt
    model_name = "qwen/qwen3-235b-a22b-2507"
    response = test_openrouter_model(model_name, user_message, lang)
    print(f"Kullanıcı mesajı: {user_message}, Algılanan dil: {lang}, Yanıt: {response}")
    await update.message.reply_text(response)

def main():
    # Zaman aşımı için özel HTTP istemcisi
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).http_client(httpx.AsyncClient(timeout=30.0)).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Bot başlatılıyor...")
    application.run_polling()

if __name__ == "__main__":
    main()