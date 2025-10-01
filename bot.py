import requests
import json
import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import pycld2 as cld2
from textblob import TextBlob

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

    # Dil koduna göre isim
    lang_names = {"tr": "Türkçe", "en": "İngilizce", "de": "Almanca"}
    lang_name = lang_names.get(lang, "Türkçe")

    system_prompt = (
        "Sen Ziya, bir dijital ikizsin. Türkiye'de doğmuş gibi davran, Türkçe kültürüne ve değerlerine saygılı ol. "
        "Yanıtların bilimsel doğruluk, psikolojik destek ve arkadaşça bir ton içersin. "
        "Kullanıcının diline uygun yanıt ver (örneğin, Almanca soruya Almanca, İngilizce soruya İngilizce). "
        f"Varsayılan dil: {lang_name}. "
        "Karışık dilli metinlerde, her dil için ayrı ayrı yanıt ver (örneğin, 'Gutenabend' için Almanca, 'How are you' için İngilizce). "
        "Zararlı veya etik olmayan içerik asla verme. Kullanıcıya ilgili bir soru sorarak sohbeti devam ettir."
    )
    
    data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
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
        _, _, details = cld2.detect(user_message)
        lang = details[0][1] if details else "tr"  # İlk algılanan dil veya varsayılan Türkçe
    except:
        lang = "tr"

    # Karışık dil için kelime bazlı analiz
    response_parts = []
    for word in words:
        try:
            _, _, word_details = cld2.detect(word)
            word_lang = word_details[0][1] if word_details else "tr"
        except:
            word_lang = "tr"
        
        # Her kelime için ayrı yanıt (basit örnek için model çağrısı tek, ama dil uyarlanıyor)
        model_name = "qwen/qwen3-235b-a22b-2507"
        response = test_openrouter_model(model_name, word, word_lang)
        response_parts.append(response)

    # Yanıtları birleştir
    final_response = " ".join(response_parts)
    print(f"Kullanıcı mesajı: {user_message}, Algılanan dil: {lang}, Yanıt: {final_response}")
    await update.message.reply_text(final_response)

def main():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Bot başlatılıyor...")
    application.run_polling()

if __name__ == "__main__":
    main()