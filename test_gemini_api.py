#!/usr/bin/env python3
import os
import sys
import time

print("Python versiyonu:", sys.version)
print("API Anahtarı kontrol ediliyor...")

# API anahtarını kontrol et
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("HATA: GOOGLE_API_KEY çevre değişkeni bulunamadı!")
    sys.exit(1)
else:
    print(f"API Anahtarı bulundu: {api_key[:5]}...{api_key[-5:]}")

try:
    print("\nGoogle Generative AI modülü import ediliyor...")
    import google.generativeai as genai
    print("Import başarılı!")
    
    print("\nGemini API'ye bağlanmaya çalışılıyor...")
    genai.configure(api_key=api_key)
    
    # Liste kullanılabilir modelleri
    print("\nKullanılabilir modeller listeleniyor...")
    models = genai.list_models()
    for model in models:
        print(f"- {model.name}")
    
    print("\nBasit bir test isteği gönderiliyor...")
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content("Merhaba, bu bir test mesajıdır.")
    
    print("\nAPI Yanıtı:")
    print(response.text)
    
    print("\nAPI testi başarılı!")
    sys.exit(0)
    
except Exception as e:
    print(f"\nHATA: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
