# PDF'den Fine-Tuning Veri Seti Oluşturma Aracı

Bu araç, PDF dosyalarından fine-tuning için veri seti oluşturan bir Python programıdır. Google Gemini 2.0 Flash modeli kullanılarak PDF içeriğinden soru-cevap çiftleri oluşturulur.

## Özellikler

- PDF dosyalarını otomatik tespit etme
- Çoklu PDF dosyalarını toplu işleme
- Sayfa sayısı fazla olan PDF'leri parçalı (batch) işleme
- Farklı PDF'lerden oluşturulan veri setlerini tek bir CSV'de birleştirme
- CSV veya JSON formatında çıktı oluşturma
- Büyük dosyalarla çalışabilme (Git LFS desteği)

## Kurulum

1. Repository'yi klonlayın:
```bash
git clone https://github.com/namruni/agemfinetuning2.git
cd agemfinetuning2
```

2. Gerekli paketleri yükleyin:
```bash
pip install pymupdf google-generativeai
```

3. Google API anahtarınızı ortam değişkeni olarak ayarlayın:
```bash
export GOOGLE_API_KEY="sizin-api-anahtariniz"
```

## Kullanım

### Temel Kullanım

```bash
python pdf_to_finetune_dataset.py --pdf dosya.pdf
```

### Birden Fazla PDF İşleme

```bash
python pdf_to_finetune_dataset.py --pdf-dir PDFs/ --all
```

### Parametreler

- `--pdf`: İşlenecek PDF dosyasının yolu
- `--pdf-dir`: İçinde PDF dosyaları bulunan dizin
- `--all`: Bulunan tüm PDF dosyalarını işle
- `--output`: Çıktı dosyalarının öneki
- `--output-dir`: Çıktıların kaydedileceği dizin
- `--no-merge`: Tüm batch'leri tek bir dosyada birleştirme
- `--merge-all`: Farklı PDF'lerden elde edilen veri setlerini tek bir CSV'de birleştir
- `--api-key`: Google API anahtarı
- `--model`: Kullanılacak model (varsayılan: gemini-2.0-flash)
- `--batch-size`: Her batch'te kaç sayfa işleneceği (varsayılan: 5)
- `--questions`: Her sayfa için kaç soru üretileceği (varsayılan: 15)
- `--format`: Çıktı formatı (csv veya json, varsayılan: csv)
- `--temperature`: Model yaratıcılık seviyesi (0.0-1.0, varsayılan: 0.7)

### Farklı PDF'lerden oluşturulan veri setlerini birleştirme

Farklı PDF'lerden elde edilen _all.csv dosyalarını tek bir CSV'de birleştirmek için:

```bash
python pdf_to_finetune_dataset.py --merge-all
```

## Dosya Yapısı

```
├── pdf_to_finetune_dataset.py   # Ana program
├── kitap1/                      # PDF'den oluşturulan veri seti klasörü
├── kitap2/                      # PDF'den oluşturulan veri seti klasörü
├── sosyal6/                     # PDF'den oluşturulan veri seti klasörü
└── birlestirilmis_ciktilar/     # Birleştirilmiş veri setleri klasörü
```

## Not

Büyük PDF dosyaları (>50MB) Git LFS ile yönetilmektedir.
