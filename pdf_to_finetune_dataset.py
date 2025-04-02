#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF'den Fine-Tuning Veri Seti Oluşturma Aracı
Bir PDF dosyasından fine-tuning için veri seti oluşturan program.
Gemini 2.0 Flash modeli kullanılarak soru-cevap çiftleri oluşturulur.
"""

import argparse
import os
import json
import csv
import time
from typing import List, Dict, Any, Optional
import fitz as pymupdf  # PyMuPDF kütüphanesi
import google.generativeai as genai  # Google Generative AI için

# Yapılandırma
class Config:
    """Program yapılandırması"""
    # Varsayılan değerler
    DEFAULT_BATCH_SIZE = 5  # Her batch'te kaç sayfa işleneceği
    DEFAULT_QUESTIONS_PER_PAGE = 15  # Her sayfa için kaç soru üretileceği
    DEFAULT_MODEL = "gemini-2.0-flash"  # Kullanılacak model
    DEFAULT_OUTPUT_FORMAT = "csv"  # Çıktı formatı
    
    def __init__(self, 
                 api_key: str = None,
                 model: str = DEFAULT_MODEL,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 questions_per_page: int = DEFAULT_QUESTIONS_PER_PAGE,
                 output_format: str = DEFAULT_OUTPUT_FORMAT,
                 temperature: float = 0.7):
        """
        Args:
            api_key: Google API anahtarı
            model: Kullanılacak model (gemini-2.0-flash, gemini-1.5-pro, vb.)
            batch_size: Her batch'te kaç sayfa işleneceği
            questions_per_page: Her sayfa için kaç soru üretileceği
            output_format: Çıktı formatı (csv veya json)
            temperature: Model yaratıcılık seviyesi (0.0-1.0)
        """
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.questions_per_page = questions_per_page
        self.output_format = output_format
        self.temperature = temperature


class PDFProcessor:
    """PDF dosyasını işleyip metin çıkaran sınıf"""
    
    def __init__(self, config: Config):
        """
        Args:
            config: Program yapılandırması
        """
        self.config = config
    
    def convert_pdf_to_text(self, pdf_path: str) -> List[str]:
        """PDF dosyasını sayfa sayfa metne çevirir
        
        Args:
            pdf_path: PDF dosyasının yolu
            
        Returns:
            Her bir sayfa için metin listesi
        """
        print(f"DEBUG: PDF dosyası metne dönüştürülmeye başlanıyor: {pdf_path}")
        if not os.path.exists(pdf_path):
            print(f"HATA: PDF dosyası bulunamadı: {pdf_path}")
            raise FileNotFoundError(f"PDF dosyası bulunamadı: {pdf_path}")
        
        try:
            print(f"DEBUG: pymupdf ile PDF açılıyor: {pdf_path}")
            doc = pymupdf.open(pdf_path)
            print(f"Belge toplam {doc.page_count} sayfa içeriyor")
        except Exception as e:
            print(f"HATA: PDF açılırken sorun oluştu: {e}")
            raise IOError(f"PDF açılırken hata oluştu: {pdf_path}. Hata: {e}")
        
        page_texts = []
        print("DEBUG: Sayfalar metne dönüştürülüyor...")
        try:
            for page_num in range(doc.page_count):
                print(f"DEBUG: Sayfa {page_num+1}/{doc.page_count} işleniyor...")
                page = doc[page_num]
                text = page.get_text()
                # Çok uzun metinlerin uzunluğunu yazdırmayı engelle
                print(f"DEBUG: Sayfa {page_num+1} metin uzunluğu: {len(text)} karakter")
                page_texts.append(text)
        except Exception as e:
            print(f"HATA: Sayfa {page_num+1} metne dönüştürülürken sorun oluştu: {e}")
            raise
            
        print(f"DEBUG: Toplam {len(page_texts)} sayfa metne çevrildi.")
        return page_texts


class QAGenerator:
    """Metin içeriğinden soru-cevap çiftleri oluşturan sınıf"""
    
    def __init__(self, config: Config):
        """
        Args:
            config: Program yapılandırması
        """
        self.config = config
        
        # API anahtarını ayarla
        if config.api_key:
            genai.configure(api_key=config.api_key)
        else:
            # Çevre değişkeninden al
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("API anahtarı sağlanmadı. Ya config ile ya da GOOGLE_API_KEY çevre değişkeni ile belirtilmeli.")
            genai.configure(api_key=api_key)
    
    def generate_qa_pairs(self, text: str) -> List[Dict[str, str]]:
        """Metinden soru-cevap çiftleri oluşturur
        
        Args:
            text: İçerik metni
            
        Returns:
            Soru-cevap çiftleri listesi
        """
        print(f"DEBUG: Gemini modeli kullanılarak soru-cevap çiftleri oluşturuluyor")
        print(f"DEBUG: Kullanılan model: {self.config.model}, sıcaklık: {self.config.temperature}")
        # Zengin sistem talimatı
        system_instruction = """
# Görevin:
Sen yüksek kaliteli çeşitli soru-cevap çiftleri oluşturmakla görevlendirilmiş bir yapay zekasın.
Görevin, verilen metinden insanların sorabileceği çeşitli türlerde sorular üretmek ve bunlara kapsamlı yanıtlar sağlamaktır.

# Talimatlar:
Verilen metin için yüksek kaliteli soru-cevap çiftleri oluştur.

## Şunları sağla:
### Dil Tutarlılığı: Sorular ve cevaplar verilen metnin diliyle aynı olmalıdır.
Örneğin eğer metin Türkçe ise, sorular ve cevaplar da Türkçe olmalıdır.

### Soru Çeşitliliği: Aşağıdaki gibi farklı türde sorular dahil et:
- Olgusal: Belirli bilgileri soran doğrudan sorular (ör. "X ne anlama gelir?")
- Kavramsal: İçeriğin ardındaki fikirleri araştıran sorular (ör. "X neden önemlidir?")
- Bağlamsal: Konunun daha geniş bağlamı veya arka planı hakkında sorular (ör. "X hangi bağlamda belirtilmiştir?")
- Nedensel: Sebepleri veya nedenleri soran sorular (ör. "X'e ne sebep olur?")
- Yöntemsel: Süreçlere veya adımlara odaklanan sorular (ör. "X nasıl başarılır?")
- Analitik: Karşılaştıran, karşıt veya değerlendiren sorular (ör. "X, Y ile nasıl karşılaştırılır?")
- Varsayımsal: Hayal edilen senaryolara dayalı sorular (ör. "X olursa ne olur?")
- Yansıtıcı: Sonuçlar veya etkiler hakkında sorular (ör. "X'in hangi etkileri vardır?")
- Spekülatif: Uygun olduğunda fikir tabanlı veya keşifsel sorular (ör. "Neden birisi X'e katılmayabilir?")
- Listeleme: Bir konuyla ilgili öğelerin, adımların veya unsurların bir listesini isteyen sorular (ör. "X'in temel unsurları nelerdir?")
- Özetleme: Bir konunun kısa bir özetini veya ana noktalarını soran sorular (ör. "X'ten çıkarılacak ana ders nedir?")

### Zorluk Dengesi: Basit, orta ve karmaşık soruları dahil et.

### Yanıt Hassasiyeti: Sağlanan içeriğe dayalı özlü, doğru yanıtlar ver.
Ancak, yanıtların çok kısa olmamasını sağla; cevabı açıkça açıklamak için yeterli bağlam verirken hala özlü olmalılar.
"Evet" veya "Hayır" gibi aşırı kısa yanıtlardan kaçın.
Yanıt, bilginin ardındaki gerekçeyi anlamak için gerekli bağlamı sağlamalıdır.

### Bağlam Farkındalığı: Tüm soruların verilen metnin içeriğine derin bir şekilde kök saldığından ve materyalin amacını ve nüanslarını anladığından emin ol.

### Tekrardan Kaçınma: Tüm soruların belirgin ve benzersiz olduğundan emin ol.
"""
        
        # Gemini modeli yapılandırması
        generation_config = {
            "temperature": self.config.temperature,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 4000,
        }
        
        # Model oluştur
        model = genai.GenerativeModel(
            model_name=self.config.model,
            generation_config=generation_config,
            system_instruction=system_instruction
        )
        
        prompt = f"""
        Aşağıdaki metin için yüksek kaliteli {self.config.questions_per_page} soru-cevap çifti oluştur.
        Her çiftin soru, cevap ve soru_türü alanlarını içermesini sağla.
        Çeşitli türlerde sorular oluştur.
        Önce metnin dilini belirle, sonra soru-cevap çiftlerini aynı dilde hazırla.
        --------------------------------
        Metin: {text}
        Soru-cevap çifti sayısı: {self.config.questions_per_page}
        --------------------------------
        Yanıtını doğrudan JSON formatında ver. Markdown işaretlerini kullanma (```json gibi). Başka açıklama ekleme, sadece düz JSON ver.
        Şu formatta olmalı:
        [
          {{"soru": "...", "cevap": "...", "soru_türü": "..."}},
          {{"soru": "...", "cevap": "...", "soru_türü": "..."}}
        ]
        """
        
        # Maksimum 3 deneme
        for attempt in range(3):
            try:
                response = model.generate_content(prompt)
                result = response.text.strip()
                
                # JSON yanıtını işle
                try:
                    # Markdown kod bloğu olarak döndürme durumunu kontrol et
                    cleaned_result = result
                    
                    # Markdown kod bloğu işaretleyicilerini temizle
                    if cleaned_result.startswith('```'):
                        # Başlangıçtaki işaretleyicileri bul
                        first_newline = cleaned_result.find('\n')
                        if first_newline != -1:
                            cleaned_result = cleaned_result[first_newline + 1:]
                        
                        # Sondaki işaretleyicileri bul
                        if '```' in cleaned_result:
                            cleaned_result = cleaned_result[:cleaned_result.rindex('```')]
                    
                    # Ekstra boşlukları temizle
                    cleaned_result = cleaned_result.strip()
                    
                    print(f"Temizlenmiş JSON: {cleaned_result[:50]}...")
                    
                    qa_pairs = json.loads(cleaned_result)
                    # Formatlama
                    standardized_pairs = []
                    for pair in qa_pairs:
                        # Alan adlarını standartlaştır
                        standardized_pair = {
                            "question": pair.get("soru", pair.get("question", "")),
                            "answer": pair.get("cevap", pair.get("answer", "")),
                            "question_type": pair.get("soru_türü", pair.get("question_type", ""))
                        }
                        standardized_pairs.append(standardized_pair)
                    return standardized_pairs
                except json.JSONDecodeError as e:
                    print(f"JSON çözümleme hatası (Deneme {attempt+1}/3): {e}")
                    print(f"Alınan yanıt: {result[:100]}...")
                    if attempt == 2:  # Son deneme
                        raise
                    time.sleep(2)  # Tekrar denemeden önce bekle
                    
            except Exception as e:
                print(f"Hata oluştu: {e} (Deneme {attempt+1}/3)")
                if attempt == 2:  # Son deneme
                    raise
                time.sleep(2)  # Tekrar denemeden önce bekle
        
        return []  # Tüm denemeler başarısız olursa boş liste döndür
    
    def process_batch(self, page_texts: List[str], start_index: int) -> List[Dict[str, str]]:
        """Bir grup sayfayı işler ve soru-cevap çiftleri oluşturur
        
        Args:
            page_texts: İşlenecek sayfa metinleri listesi
            start_index: Sayfaların başlangıç indeksi
            
        Returns:
            Oluşturulan soru-cevap çiftleri listesi
        """
        all_qa_pairs = []
        
        for i, page_text in enumerate(page_texts):
            page_num = start_index + i
            print(f"Sayfa {page_num+1} işleniyor...")
            
            qa_pairs = self.generate_qa_pairs(page_text)
            if qa_pairs:
                # Her çifte sayfa numarası ekle
                for pair in qa_pairs:
                    pair["page"] = page_num + 1
                
                all_qa_pairs.extend(qa_pairs)
                print(f"Sayfa {page_num+1} için {len(qa_pairs)} soru-cevap çifti oluşturuldu")
            else:
                print(f"Sayfa {page_num+1} için soru-cevap çifti oluşturulamadı")
        
        return all_qa_pairs


class OutputManager:
    """Soru-cevap çiftlerini dosyaya kaydeden sınıf"""
    
    def __init__(self, config: Config):
        """
        Args:
            config: Program yapılandırması
        """
        self.config = config
        
    def merge_multiple_files(self, file_paths: List[str], output_prefix: str) -> None:
        """Birden fazla çıktı dosyasını tek bir dosyada birleştirir
        
        Args:
            file_paths: Birleştirilecek dosya yolları listesi
            output_prefix: Çıktı dosyasının öneki
        """
        if self.config.output_format.lower() == "csv":
            self._merge_multiple_csv_files(file_paths, output_prefix)
        else: # json
            self._merge_multiple_json_files(file_paths, output_prefix)
            
    def _merge_multiple_csv_files(self, file_paths: List[str], output_prefix: str) -> None:
        """Birden fazla CSV dosyasını tek bir dosyada birleştirir
        
        Args:
            file_paths: Birleştirilecek CSV dosya yolları listesi
            output_prefix: Çıktı dosyasının öneki
        """
        all_rows = []
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    csv_reader = csv.DictReader(file)
                    
                    # Dosya adını kaynak olarak ekle
                    book_name = os.path.splitext(os.path.basename(file_path))[0]
                    
                    for row in csv_reader:
                        # Kitap bilgisini ekle
                        row['kaynak'] = book_name
                        all_rows.append(row)
            except Exception as e:
                print(f"Uyarı: {file_path} dosyası birleştirilemedi: {e}")
        
        if all_rows:
            output_file = f"{output_prefix}.csv"
            with open(output_file, 'w', encoding='utf-8', newline='') as file:
                # Tüm alanları belirle (kaynak alanını da ekleyerek)
                # Tüm satırlardaki mevcut alanları belirle
                all_fields = set()
                for row in all_rows:
                    all_fields.update(row.keys())
                
                # Önemli alanların sırasını korumak için sıralı alanlar listesi
                # Öncelikli alanlar
                priority_fields = ['question', 'answer', 'question_type', 'page', 'kaynak']
                
                # Son alan listesi - önce öncelikli alanlar, sonra diğerleri
                fieldnames = [f for f in priority_fields if f in all_fields]
                
                # Kalan alanları ekle
                for field in all_fields:
                    if field not in fieldnames:
                        fieldnames.append(field)
                
                csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
                csv_writer.writeheader()
                csv_writer.writerows(all_rows)
                
            print(f"Tüm kitaplardan {len(all_rows)} soru-cevap çifti {output_file} dosyasına birleştirildi.")
        else:
            print("Birleştirilecek veri bulunamadı.")
            
    def _merge_multiple_json_files(self, file_paths: List[str], output_prefix: str) -> None:
        """Birden fazla JSON dosyasını tek bir dosyada birleştirir
        
        Args:
            file_paths: Birleştirilecek JSON dosya yolları listesi
            output_prefix: Çıktı dosyasının öneki
        """
        all_qa_pairs = []
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    qa_pairs = json.load(file)
                    
                    # Dosya adını kaynak olarak ekle
                    book_name = os.path.splitext(os.path.basename(file_path))[0]
                    
                    for qa_pair in qa_pairs:
                        # Kitap bilgisini ekle
                        qa_pair['kaynak'] = book_name
                        all_qa_pairs.append(qa_pair)
            except Exception as e:
                print(f"Uyarı: {file_path} dosyası birleştirilemedi: {e}")
        
        if all_qa_pairs:
            output_file = f"{output_prefix}.json"
            with open(output_file, 'w', encoding='utf-8') as file:
                json.dump(all_qa_pairs, file, ensure_ascii=False, indent=2)
                
            print(f"Tüm kitaplardan {len(all_qa_pairs)} soru-cevap çifti {output_file} dosyasına birleştirildi.")
        else:
            print("Birleştirilecek veri bulunamadı.")
    
    def save_to_csv(self, qa_pairs: List[Dict[str, str]], filename: str) -> None:
        """Soru-cevap çiftlerini CSV dosyasına kaydeder
        
        Args:
            qa_pairs: Soru-cevap çiftleri listesi
            filename: Dosya adı (uzantısız)
        """
        if not qa_pairs:
            print(f"Uyarı: Kaydedilecek soru-cevap çifti yok: {filename}.csv")
            return
            
        output_file = f"{filename}.csv"
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['question', 'answer', 'question_type', 'page']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for item in qa_pairs:
                writer.writerow(item)
        
        print(f"{len(qa_pairs)} soru-cevap çifti {output_file} dosyasına kaydedildi")
    
    def save_to_json(self, qa_pairs: List[Dict[str, str]], filename: str) -> None:
        """Soru-cevap çiftlerini JSON dosyasına kaydeder
        
        Args:
            qa_pairs: Soru-cevap çiftleri listesi
            filename: Dosya adı (uzantısız)
        """
        if not qa_pairs:
            print(f"Uyarı: Kaydedilecek soru-cevap çifti yok: {filename}.json")
            return
            
        output_file = f"{filename}.json"
        
        with open(output_file, 'w', encoding='utf-8') as jsonfile:
            json.dump(qa_pairs, jsonfile, ensure_ascii=False, indent=2)
        
        print(f"{len(qa_pairs)} soru-cevap çifti {output_file} dosyasına kaydedildi")
    
    def save_output(self, qa_pairs: List[Dict[str, str]], filename: str) -> None:
        """Belirlenen formatta çıktıyı kaydeder
        
        Args:
            qa_pairs: Soru-cevap çiftleri listesi
            filename: Dosya adı (uzantısız)
        """
        if self.config.output_format.lower() == 'json':
            self.save_to_json(qa_pairs, filename)
        else:
            self.save_to_csv(qa_pairs, filename)
    
    def merge_batch_files(self, output_prefix: str, batch_count: int) -> None:
        """Batch dosyalarını tek bir dosyada birleştirir
        
        Args:
            output_prefix: Dosya öneki
            batch_count: Toplam batch sayısı
        """
        merged_filename = f"{output_prefix}_all"
        
        # Tüm veri setini saklamak için liste
        all_qa_pairs = []
        
        if self.config.output_format.lower() == 'json':
            # JSON dosyalarını birleştir
            for i in range(1, batch_count + 1):
                batch_file = f"{output_prefix}_batch_{i}.json"
                try:
                    with open(batch_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        all_qa_pairs.extend(data)
                except Exception as e:
                    print(f"Uyarı: {batch_file} dosyası okunamadı: {e}")
            
            # Birleştirilmiş JSON'ı kaydet
            self.save_to_json(all_qa_pairs, merged_filename)
            
        else:  # CSV dosyaları için
            # İlk dosyadan başlıkları al
            with open(f"{output_prefix}_batch_1.csv", 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader)  # İlk satır başlıklar
            
            # Birleştirilmiş CSV'yi oluştur
            with open(f"{merged_filename}.csv", 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(headers)  # Başlıkları yaz
                
                # Tüm batch dosyalarını oku ve birleştir
                for i in range(1, batch_count + 1):
                    batch_file = f"{output_prefix}_batch_{i}.csv"
                    try:
                        with open(batch_file, 'r', newline='', encoding='utf-8') as f:
                            reader = csv.reader(f)
                            next(reader)  # Başlık satırını atla
                            for row in reader:
                                writer.writerow(row)
                                all_qa_pairs.append(row)  # Satır sayısını saymak için
                    except Exception as e:
                        print(f"Uyarı: {batch_file} dosyası okunamadı: {e}")
        
        print(f"\nTüm batch'ler birleştirildi: {len(all_qa_pairs)} soru-cevap çifti {merged_filename}.{self.config.output_format} dosyasına kaydedildi")
        
    def merge_all_dataset_files(self) -> None:
        """Farklı PDF'lerden elde edilen CSV dosyalarını birleştirir.
        Kullanıcıya hangi dosyaları birleştirmek istediğini sorar.
        """
        # _all.csv veya _all.json dosyalarını bul
        import glob
        
        all_files = []
        
        # CSV veya JSON formatında tüm *_all.* dosyalarını bul
        extension = "." + self.config.output_format.lower()
        pattern = f"*_all{extension}"
        all_files = glob.glob(pattern)
        
        if not all_files:
            print(f"Birleştirilecek dosya bulunamadı. (Aranan: {pattern})")
            return
        
        print(f"\n{len(all_files)} birleştirilebilir dosya bulundu:")
        for i, file in enumerate(all_files):
            print(f"{i+1}. {file}")
        
        print("\nBirleştirmek istediğiniz dosyaları seçin:")
        print("1. Tüm dosyaları birleştir")
        print("2. Belirli dosyaları seç")
        print("3. İptal et")
        
        try:
            choice = int(input("Seçiminiz (1-3): "))
            
            if choice == 1:
                # Tüm dosyaları birleştir
                selected_files = all_files
            elif choice == 2:
                # Belirli dosyaları seç
                print("\nBirleştirmek istediğiniz dosyaları seçin (1,3,5 gibi numaraları virgülle ayırın):")
                for i, file in enumerate(all_files):
                    print(f"{i+1}. {file}")
                    
                indices_input = input("Seçimleriniz: ")
                selected_indices = [int(idx.strip()) - 1 for idx in indices_input.split(',')]
                
                selected_files = []
                for idx in selected_indices:
                    if 0 <= idx < len(all_files):
                        selected_files.append(all_files[idx])
                    else:
                        print(f"Uyarı: {idx+1} geçerli bir seçim değil, atlanıyor.")
            elif choice == 3:
                print("İşlem iptal edildi.")
                return
            else:
                print("Geçersiz seçim, işlem iptal edildi.")
                return
            
            if not selected_files:
                print("Seçilen dosya yok, işlem iptal edildi.")
                return
            
            # Çıktı dosya adını belirle
            output_name = input("\nBirleştirilmiş dosya için isim girin (uzantı olmadan): ")
            if not output_name.strip():
                output_name = "combined_dataset"
            
            # Dosyaları birleştir
            print(f"\n{len(selected_files)} dosya birleştiriliyor...")
            self.merge_multiple_files(selected_files, output_name)
            print(f"Birleştirme işlemi tamamlandı. Dosya: {output_name}.{self.config.output_format}")
            
        except ValueError:
            print("Geçersiz giriş, işlem iptal edildi.")
        except Exception as e:
            print(f"Hata oluştu: {e}")
            import traceback
            traceback.print_exc()


class FineTuneDatasetGenerator:
    """PDF'den fine-tune veri seti oluşturan ana sınıf"""
    
    def __init__(self, config: Config):
        """
        Args:
            config: Program yapılandırması
        """
        self.config = config
        self.pdf_processor = PDFProcessor(config)
        self.qa_generator = QAGenerator(config)
        self.output_manager = OutputManager(config)
    
    def generate_dataset(self, pdf_path: str, output_prefix: Optional[str] = None) -> None:
        """PDF'den fine-tune veri seti oluşturur
        
        Args:
            pdf_path: PDF dosyasının yolu
            output_prefix: Çıktı dosyalarının öneki (belirtilmezse PDF adı kullanılır)
        """
        # Hata ayıklama logu ekle
        print(f"DEBUG: generate_dataset başladı - pdf_path: {pdf_path}, output_prefix: {output_prefix}")
        # PDF'i metin olarak oku
        try:
            print("PDF metni çıkarılıyor...")
            page_texts = self.pdf_processor.convert_pdf_to_text(pdf_path)
            print(f"DEBUG: PDF'den {len(page_texts)} sayfa metin çıkarıldı")
        except Exception as e:
            print(f"HATA: PDF metni çıkarılırken sorun oluştu: {str(e)}")
            raise
        
        # Çıktı dosyası öneğini belirle
        if not output_prefix:
            output_prefix = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # Sayfaları batch'ler halinde işle
        batch_size = self.config.batch_size
        batch_count = 0
        
        for batch_index, batch_start in enumerate(range(0, len(page_texts), batch_size)):
            batch_number = batch_index + 1
            batch_count = batch_number  # Son batch numarasını sakla
            print(f"\nBatch {batch_number} başlıyor...")
            
            # Batch için sayfaları al
            batch_end = min(batch_start + batch_size, len(page_texts))
            batch_pages = page_texts[batch_start:batch_end]
            
            # Soru-cevap çiftleri oluştur
            qa_pairs = self.qa_generator.process_batch(batch_pages, batch_start)
            
            # Sonuçları kaydet
            batch_filename = f"{output_prefix}_batch_{batch_number}"
            self.output_manager.save_output(qa_pairs, batch_filename)
            
            print(f"Batch {batch_number} tamamlandı")
        
        # Tüm batch'leri tek bir dosyada birleştir
        print("\nTüm batch'ler birleştiriliyor...")
        self.output_manager.merge_batch_files(output_prefix, batch_count)
        
        print(f"\nİşlem tamamlandı. {len(page_texts)} sayfa işlendi.")


def find_pdf_files(sort_alphabetically=True):
    """Mevcut dizinde veya alt dizinlerde PDF dosyalarını arar
    
    Args:
        sort_alphabetically: PDF dosyalarını alfabetik sıraya göre sırala
        
    Returns:
        PDF dosya yollarının listesi
    """
    import glob
    import os
    
    # Önce mevcut dizindeki PDF'leri ara
    pdf_files = glob.glob("*.pdf")
    
    # Eğer bulunamadıysa alt dizinlere bak
    if not pdf_files:
        pdf_files = glob.glob("**/*.pdf", recursive=True)
    
    # Mutlak yolları al
    pdf_files = [os.path.abspath(pdf) for pdf in pdf_files]
    
    # Alfabetik sıralama (isteğe bağlı)
    if sort_alphabetically and pdf_files:
        pdf_files.sort()
    
    return pdf_files

def main():
    """Ana program"""
    # Program başlangıcı mesajı
    print("\n============================================")
    print("PDF'den Fine-Tuning Veri Seti Oluşturma Aracı")
    print("============================================\n") 
    # API anahtarını kontrol et
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("UYARI: GOOGLE_API_KEY çevre değişkeni bulunamadı.")
        print("Ya --api-key parametresi kullanın ya da GOOGLE_API_KEY çevre değişkenini ayarlayın.")
    # Komut satırı argümanlarını tanımla
    parser = argparse.ArgumentParser(description="PDF'den fine-tuning için veri seti oluşturur")
    
    parser.add_argument("--pdf", dest="pdf_path", help="İşlenecek PDF dosyasının yolu (belirtilmezse otomatik bulunur)")
    parser.add_argument("--pdf-dir", dest="pdf_directory", help="İçinde PDF dosyaları bulunan dizin (birden fazla kitap işlemek için)")
    parser.add_argument("--all", action="store_true", help="Bulunan tüm PDF dosyalarını işle")
    parser.add_argument("--output", "-o", help="Çıktı dosyalarının öneki")
    parser.add_argument("--output-dir", help="Çıktıların kaydedileceği dizin")
    parser.add_argument("--no-merge", action="store_true", default=False, 
                      help="Tüm kitapların çıktılarını tek bir CSV dosyasında birleştirme")
    parser.add_argument("--merge-all", action="store_true", default=False, 
                      help="Farklı PDF'lerden elde edilen _all.csv dosyalarını tek bir dosyada birleştir")
    parser.add_argument("--api-key", help="Google API anahtarı (belirtilmezse GOOGLE_API_KEY çevre değişkeni kullanılır)")
    parser.add_argument("--model", default=Config.DEFAULT_MODEL, 
                      help=f"Kullanılacak model (varsayılan: {Config.DEFAULT_MODEL})")
    parser.add_argument("--batch-size", type=int, default=Config.DEFAULT_BATCH_SIZE,
                      help=f"Her batch'te kaç sayfa işleneceği (varsayılan: {Config.DEFAULT_BATCH_SIZE})")
    parser.add_argument("--questions", type=int, default=Config.DEFAULT_QUESTIONS_PER_PAGE,
                      help=f"Her sayfa için kaç soru üretileceği (varsayılan: {Config.DEFAULT_QUESTIONS_PER_PAGE})")
    parser.add_argument("--format", choices=["csv", "json"], default=Config.DEFAULT_OUTPUT_FORMAT,
                      help=f"Çıktı formatı (varsayılan: {Config.DEFAULT_OUTPUT_FORMAT})")
    parser.add_argument("--temperature", type=float, default=0.7,
                      help="Model yaratıcılık seviyesi (0.0-1.0, varsayılan: 0.7)")
    
    args = parser.parse_args()
    
    # Çıktı dizinini oluştur
    output_dir = args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # PDF dosyalarını belirle
    pdf_files_to_process = []
    
    # Tek PDF yolu belirtilmişse
    if args.pdf_path:
        pdf_files_to_process.append(args.pdf_path)
    
    # PDF dizini belirtilmişse
    elif args.pdf_directory:
        if os.path.isdir(args.pdf_directory):
            dir_pdf_files = [os.path.join(args.pdf_directory, f) for f in os.listdir(args.pdf_directory) 
                             if f.lower().endswith('.pdf')]
            if dir_pdf_files:
                if args.all:
                    pdf_files_to_process.extend(dir_pdf_files)
                else:
                    print(f"{args.pdf_directory} dizininde {len(dir_pdf_files)} PDF bulundu.")
                    print("Tüm PDF'leri işlemek için --all parametresini kullanın veya işlemek istediğiniz dosyayı seçin:")
                    for i, pdf in enumerate(dir_pdf_files):
                        print(f"{i+1}. {os.path.basename(pdf)}")
                    try:
                        choices = input("Seçimleriniz (1,3,5 gibi numaraları virgülle ayırın veya 'hepsi' yazın): ")
                        if choices.lower() == 'hepsi':
                            pdf_files_to_process.extend(dir_pdf_files)
                        else:
                            indices = [int(idx.strip()) - 1 for idx in choices.split(',')]
                            for idx in indices:
                                if 0 <= idx < len(dir_pdf_files):
                                    pdf_files_to_process.append(dir_pdf_files[idx])
                                else:
                                    print(f"Uyarı: {idx+1} geçerli bir seçim değil, atlanıyor.")
                    except ValueError:
                        print("Geçersiz giriş, işlem iptal ediliyor.")
                        return 1
            else:
                print(f"Hata: {args.pdf_directory} dizininde PDF dosyası bulunamadı.")
                return 1
        else:
            print(f"Hata: {args.pdf_directory} geçerli bir dizin değil.")
            return 1
    
    # PDF belirtilmemişse, otomatik bul
    else:
        print("PDF dosyası otomatik olarak aranıyor...")
        auto_pdf_files = find_pdf_files()
        print(f"DEBUG: {len(auto_pdf_files)} PDF dosyası bulundu: {auto_pdf_files}")
        if not auto_pdf_files:
            print("Hata: Herhangi bir PDF dosyası bulunamadı. Lütfen PDF dosyasının yolunu belirtin.")
            return 1
        
        if len(auto_pdf_files) == 1:
            pdf_files_to_process.append(auto_pdf_files[0])
            # Mutlak yolun doğruluğunu kontrol et
            if os.path.exists(auto_pdf_files[0]):
                print(f"PDF dosyası otomatik olarak bulundu: {auto_pdf_files[0]}")
            else:
                print(f"Uyarı: Bulunan PDF yolu geçerli değil: {auto_pdf_files[0]}")
                print("Lütfen tam dosya yolunu kontrol edin.")
                return 1
        else:
            print("Birden fazla PDF dosyası bulundu. Varsayılan olarak tümü sırayla işlenecek.")
            print("Bulunan PDF'ler:")
            for i, pdf in enumerate(auto_pdf_files):
                print(f"{i+1}. {pdf}")
            
            try:
                response = input("Tüm PDF'leri işlemek için Enter'a basın veya belirli dosyaların numaralarını virgülle ayırarak girin: ")
                if not response.strip() or response.lower() == 'hepsi' or args.all:
                    print("Tüm PDF'ler sırayla işlenecek...")
                    pdf_files_to_process.extend(auto_pdf_files)
                    # Varsayılan olarak birleştirme özelliği zaten etkin
                else:
                    indices = [int(idx.strip()) - 1 for idx in response.split(',') if idx.strip()]
                    for idx in indices:
                        if 0 <= idx < len(auto_pdf_files):
                            pdf_files_to_process.append(auto_pdf_files[idx])
                        else:
                            print(f"Uyarı: {idx+1} geçerli bir seçim değil, atlanıyor.")
                    # Birleştirme her zaman varsayılan olarak etkin, özellikle devre dışı bırakılmadıysa
            except ValueError:
                print("Geçersiz giriş, işlem iptal ediliyor.")
                return 1
    
    if not pdf_files_to_process:
        print("Hata: İşlenecek PDF dosyası bulunamadı.")
        return 1
    
    # Yapılandırmayı oluştur
    config = Config(
        api_key=args.api_key,
        model=args.model,
        batch_size=args.batch_size,
        questions_per_page=args.questions,
        output_format=args.format,
        temperature=args.temperature
    )
    
    # Veri seti oluşturucuyu başlat
    generator = FineTuneDatasetGenerator(config)
    
    # Sonunda birleştirilecek tüm kitap çıktıları
    all_output_files = []
    
    # Her PDF'i işle
    for i, pdf_path in enumerate(pdf_files_to_process):
        print(f"\n==============================================")
        print(f"PDF {i+1}/{len(pdf_files_to_process)} işleniyor: {pdf_path}")
        print(f"==============================================")
        
        # PDF için klasör adı oluştur
        if args.output:
            if len(pdf_files_to_process) > 1:
                # Birden fazla PDF için dosya adına indeks ekle
                folder_name = f"{args.output}_{i+1}"
            else:
                folder_name = args.output
        else:
            # PDF adını kullan
            folder_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # PDF için özel klasör oluştur
        if output_dir:
            pdf_output_dir = os.path.join(output_dir, folder_name)
        else:
            pdf_output_dir = folder_name
        
        # Klasörü oluştur
        os.makedirs(pdf_output_dir, exist_ok=True)
        print(f"PDF çıktıları için klasör oluşturuldu: {pdf_output_dir}")
        
        # Çıktı dosyasının öneki (PDF klasörü içindeki temel dosya adı)
        base_output = folder_name
        output_prefix = os.path.join(pdf_output_dir, base_output)
        
        # PDF'i işle
        try:
            generator.generate_dataset(pdf_path, output_prefix)
            
            # Son merge dosyasını all_output_files listesine ekle
            try:
                if not args.no_merge:
                    # PDF için tüm batch'ı işledikten sonraki birleştirilmiş dosya
                    final_output = f"{output_prefix}.{config.output_format}"
                    
                    # Dosyanın var olup olmadığını kontrol et
                    if os.path.exists(final_output):
                        all_output_files.append(final_output)
                    else:
                        print(f"Uyarı: {final_output} dosyası bulunamadı. Birleştirme için atlanıyor.")
            except Exception as e:
                print(f"Uyarı: Çıktı dosyası işlenirken hata oluştu: {e}")
                
        except Exception as e:
            print(f"Hata ({os.path.basename(pdf_path)}): {e}")
            continue
    
    # Tüm kitapların çıktılarını birleştir
    if not args.no_merge and len(all_output_files) > 0:
        print("\nTüm kitapların çıktıları birleştiriliyor...")
        
        # Ana çıktı klasörü oluştur
        merged_dir = "birlestirilmis_ciktilar"
        if output_dir:
            merged_dir = os.path.join(output_dir, merged_dir)
        
        # Klasörü oluştur
        os.makedirs(merged_dir, exist_ok=True)
        print(f"Birleştirilmiş çıktılar için klasör oluşturuldu: {merged_dir}")
        
        # Birleştirilmiş dosya adı
        if args.output:
            merged_output = f"{args.output}_tüm_kitaplar"
        else:
            # Eğer belirli bir dizinde işleniyorsa dizin adını kullan
            if args.pdf_directory:
                dir_name = os.path.basename(os.path.normpath(args.pdf_directory))
                merged_output = f"{dir_name}_tüm_kitaplar"
            else:
                merged_output = "tüm_kitaplar_birleşik"
        
        print(f"DEBUG: Birleştirilecek dosyalar: {valid_files}")
        
        # Birleştirilmiş çıktının tam yolu
        merged_output = os.path.join(merged_dir, merged_output)
        
                # Birleştirilecek dosya listesini kontrol et
        valid_files = [f for f in all_output_files if os.path.exists(f)]
        
        if not valid_files:
            print("Uyarı: Birleştirilecek geçerli dosya bulunamadı.")
        else:
            print(f"Birleştiriliyor: {len(valid_files)} dosya")
            # Tüm dosyaları her zaman CSV formatında birleştir (kullanıcı her format için çıktı üretebilir,
            # ama birleştirme her zaman CSV'de yapılacak)
            merge_config = Config(
                api_key=config.api_key,
                model=config.model,
                batch_size=config.batch_size,
                questions_per_page=config.questions_per_page,
                output_format="csv",  # Birleştirilen dosyayı CSV olarak kaydet
                temperature=config.temperature
            )
            
            OutputManager(merge_config).merge_multiple_files(valid_files, merged_output)
        
        print(f"Tüm kitaplar başarıyla tek bir CSV'de birleştirildi: {merged_output}.csv")
        print(f"CSV dosyası şu konumda: {os.path.abspath(merged_output)}.csv")
    
    print(f"\nTüm işlemler tamamlandı. {len(pdf_files_to_process)} PDF dosyası işlendi.")
    
    # Tüm PDF'ler işlendikten sonra, eğer --merge-all parametresi belirtilmişse
    # farklı PDF'lerden elde edilen _all dosyalarını birleştir
    if args.merge_all:
        print("\n===================================================")
        print("Farklı PDF'lerden elde edilen veri setlerini birleştirme")
        print("===================================================\n")
        output_manager = OutputManager(config)
        output_manager.merge_all_dataset_files()
    
    return 0


if __name__ == "__main__":
    try:
        print("Program başlıyor...")
        exit(main())
    except KeyboardInterrupt:
        print("\nİşlem kullanıcı tarafından durduruldu.")
        exit(1)
    except Exception as e:
        print(f"\nBeklenmeyen bir hata oluştu: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
