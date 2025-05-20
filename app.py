import os
import sys
import asyncio
import streamlit as st
import pdfplumber
import torch
import csv
import pandas as pd
import re
import nltk
import unicodedata
import requests
import uuid
from rapidfuzz import fuzz
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from sentence_transformers import SentenceTransformer, util
from transformers import BertForSequenceClassification, BertTokenizerFast 
from db_utils import get_all_cvs, get_all_kriterler, add_kriter, delete_kriter, save_cv, delete_cv, check_kriter_exists, get_all_job_descriptions, delete_job_description, add_job_description, delete_all_cvler, toggle_kriter_aktiflik

#kullanıcı
def kimlik_dogrula():
    if "auth" not in st.session_state:
        st.session_state.auth = False
    if st.session_state.auth:
        return True

    with st.form("login_form"):
        col1, col2 = st.columns(2)
        username = col1.text_input("👤 Kullanıcı Adı")
        password = col2.text_input("🔐 Şifre", type="password")
        submitted = st.form_submit_button ("Giriş Yap", type="primary")
        if submitted:
            if username == "admin" and password == "admin":
                st.session_state.auth = True
                st.success("✅ Giriş başarılı!")
                st.rerun()
            else:
                st.error("❌ Hatalı kullanıcı adı veya şifre")
    return False

# NLTK
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")


#bugfix
if sys.platform == "win32" and sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
torch.classes.__path__ = []

#gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#SBERT
sbert_model = SentenceTransformer("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr", device=device)

#BERT MUR55
bert_model = BertForSequenceClassification.from_pretrained("MUR55/bert_turkish_personality_analysis").to(device)
bert_tokenizer = BertTokenizerFast.from_pretrained("MUR55/bert_turkish_personality_analysis")

#labels
labels_url = "https://huggingface.co/MUR55/bert_turkish_personality_analysis/resolve/main/labels.txt"
response = requests.get(labels_url)
LABELS = [l.strip() for l in response.text.splitlines()]


#thresholds
THRESHOLDS = {}

thresholds_url = "https://huggingface.co/MUR55/bert_turkish_personality_analysis/resolve/main/thresholds.txt"

try:
    response = requests.get(thresholds_url)
    response.raise_for_status()
    for line in response.text.splitlines():
        if ":" in line:
            label, value = line.strip().split(":", 1)
            THRESHOLDS[label] = float(value)
except Exception as e:
    print(f"thresholds.txt indirilemedi, varsayılan eşikler kullanılacak. Hata: {e}")
    for label in LABELS:
        THRESHOLDS[label] = 0.5

#streamlit
st.set_page_config(page_title="Akıllı CV Analiz Sistemi", layout="wide")

st.title("📄 Akıllı CV Analiz Sistemi")
#tokenize 
def kelime_ayıkla(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return "\n".join([page.extract_text() or "" for page in pdf.pages])

def kisilik_tahmin(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        logits = bert_model(**inputs).logits
        probs = torch.sigmoid(logits).squeeze().tolist()
        
        if not isinstance(probs, list):
            probs = [probs]
    
    return probs

def sbert_benzerlik(cv_text, job_desc):
    if not cv_text.strip() or not job_desc.strip():
        return 0.0

    cv_text_processed =benzerlik_oncesi(cv_text)
    job_desc_processed = benzerlik_oncesi(job_desc)
    
    # preprocess sonrası da boş mu kontrolü
    if not cv_text_processed.strip() or not job_desc_processed.strip():
        return 0.0

    cv_chunks = [cv_text_processed[i:i+512] for i in range(0, len(cv_text_processed), 512)]
    job_chunks = [job_desc_processed[i:i+512] for i in range(0, len(job_desc_processed), 512)]
    
    # chunk sonrası da boş mu kontrolü
    if not cv_chunks or not job_chunks:
        return 0.0

    cv_embeddings = sbert_model.encode(cv_chunks, convert_to_tensor=True)
    job_embeddings = sbert_model.encode(job_chunks, convert_to_tensor=True)
    
    similarities = []
    for cv_emb in cv_embeddings:
        chunk_similarities = []
        for job_emb in job_embeddings:
            sim = util.pytorch_cos_sim(cv_emb, job_emb).item()
            chunk_similarities.append(sim)
        
        if chunk_similarities:  # boş listeye max uygulamadan önce koruma
            similarities.append(max(chunk_similarities))
    
    if not similarities:  # hala boşsa
        return 0.0
    
    return round(max(similarities) * 100, 2)


def keyword_benzerlik(cv_text, job_desc):
    #keywords çıkar
    job_keywords = extract_keywords_from_job(job_desc)
    
    keyword_match_scores = []
    
    for keyword in job_keywords:
        #close eşleşme
        keyword_embedding = sbert_model.encode(keyword, convert_to_tensor=True)
        cv_sentences = extract_meaningful_sentences(cv_text)
        
        best_match_score = 0
        for sentence in cv_sentences:
            sentence_embedding = sbert_model.encode(sentence, convert_to_tensor=True)
            sim_score = util.pytorch_cos_sim(keyword_embedding, sentence_embedding).item()
            best_match_score = max(best_match_score, sim_score)
        
        keyword_match_scores.append(best_match_score)
    
    #ort
    if keyword_match_scores:
        return sum(keyword_match_scores) / len(keyword_match_scores) * 100
    return 0

def extract_keywords_from_job(job_desc):

    job_desc = job_desc.lower()
    sentences = extract_meaningful_sentences(job_desc)
    #anaht
    keywords = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        pos_tags = pos_tag(words)
        
        #sıfats
        important_words = [word for word, tag in pos_tags if tag.startswith('N') or tag.startswith('ADJ')]
        keywords.extend(important_words)
    #max
    from collections import Counter
    word_counts = Counter(keywords)
    return [word for word, count in word_counts.most_common(15)]

def benzerlik_oncesi(text):
    
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text) #posta
    text = re.sub(r'\b\d{3}[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}\b', '', text)#tel
    
    sentences = extract_meaningful_sentences(text)
    sentences = temizle_cumleler(sentences)
    
    return " ".join(sentences)

def degerlendir_kriter(cv_text, threshold=85):
    kriterler = [k for k in get_all_kriterler() if k.get("aktif", True)]
    toplam_agirlik = sum([k["agirlik"] for k in kriterler]) or 1
    puan = 0
    matched = []
    unmatched = []

    cv_text_clean = cv_text.lower()
    kelimeler = [w.lower() for w in word_tokenize(cv_text_clean) if w.isalpha()]

    for k in kriterler:
        param = k["parametre"].lower()
        if fuzz.partial_ratio(param, cv_text_clean) >= threshold or any(fuzz.ratio(param, kelime) >= threshold for kelime in kelimeler):
            puan += k["agirlik"]
            matched.append(k)
        else:
            unmatched.append(k)

    score = round((puan / toplam_agirlik) * 100, 2)
    return score, matched, unmatched

def extract_meaningful_sentences(text, min_length=5, max_length=100):
    sentences = sent_tokenize(text)
    return [s for s in sentences if min_length <= len(s.split()) <= max_length]

def temizle_cumleler(cumleler):
    filtre_kelimeler = [
        "gmail", "adres", "telefon", "numarası", "istanbul", "ankara",
        "ehliyet", "cinsiyet", "doğum", "türkiye", "mail", "email",
        "hakkımda", "iletişim", "cv", "tc", "nüfus", "uyruk", 
        "tel", "cep", "gsm","instagram", "linkedin", "twitter", "facebook",
        "www.", "http", "https", "sosyal medya", "web sitesi", "blog",
    ]
    temiz = []
    for c in cumleler:
        c = c.strip()
        if len(c.split()) < 3:
            continue
        if not any(kw in c.lower() for kw in filtre_kelimeler) and not re.search(r"\d{3,}", c):
            temiz.append(c)
    return temiz

def normalize_turkish_text(text):
  
    text = unicodedata.normalize('NFKC', text)
    
    turkish_replacements = {
        "Ä±": "ı", "Ä°": "İ", "Äž": "ğ", "Ä": "ğ",
        "ÅŸ": "ş", "Å": "ş", "Ã§": "ç", "Ã¼": "ü",
        "Ã¶": "ö", "â": "a"
    }
    for wrong, correct in turkish_replacements.items():
        text = text.replace(wrong, correct)
    
    return text

#genelanaliz
def analiz_cv_kelime(full_text, job_desc=""):
    cumleler_raw = extract_meaningful_sentences(full_text)
    cumleler = temizle_cumleler_gelismis(cumleler_raw, full_text)
    
    if not cumleler:
        return {"error": "CV'den anlamlı cümle çıkarılamadı"}
    
    cv_paragraphs = chunk_text(full_text, max_length=512)
    
    predictions = []
    confidences = []
    
    for c in cumleler:
        probs = kisilik_tahmin(c)
        predictions.append(probs)
        confidence = max([abs(p-0.5)+0.5 for p in probs])
        confidences.append(confidence * 0.8) 
    
    for p in cv_paragraphs:
        probs = kisilik_tahmin(p)
        predictions.append(probs)
        
        confidence = max([abs(p-0.5)+0.5 for p in probs])
        confidences.append(confidence * 1.2)
    
    weighted_preds = []
    for pred, conf in zip(predictions, confidences):
        weighted_preds.append([p * conf for p in pred])
    
    sum_conf = sum(confidences)
    if sum_conf > 0:
        final_preds = [sum(values) / sum_conf for values in zip(*weighted_preds)]
    else:
        final_preds = [sum(values) / len(predictions) for values in zip(*predictions)]
    
    kw_adjustments = keyword_finetuning(full_text)
    
    for i, adj in enumerate(kw_adjustments):
        if i < len(final_preds):
            final_preds[i] = final_preds[i] * 0.4 + adj * 0.6

    binary_preds = []
    for i, score in enumerate(final_preds):
        label = LABELS[i]
        threshold = THRESHOLDS.get(label, 0.5)
        binary_preds.append(1 if score >= threshold else 0)
    
    paired_preds = [(LABELS[i], score) for i, score in enumerate(final_preds)]
    sorted_preds = sorted(paired_preds, key=lambda x: x[1], reverse=True)
    differentiated_preds = diff_skor(sorted_preds)
    
    top_traits = [f"{label} ({score:.2f})" for label, score in differentiated_preds[:2]]
    bottom_traits = [f"{label} ({score:.2f})" for label, score in differentiated_preds[-2:]]
    
    trait_examples = {label: [] for label in LABELS}
    
    for i, c in enumerate(cumleler):
        if i >= len(predictions):
            continue
            
        probs = predictions[i]
        if isinstance(probs, list) and probs:
            max_label_idx = probs.index(max(probs))
            max_label = LABELS[max_label_idx]
            
            if probs[max_label_idx] >= THRESHOLDS.get(max_label, 0.5):
                trait_examples[max_label].append((c, probs[max_label_idx]))
    
    for label in LABELS:
        candidates = sorted(trait_examples[label], key=lambda x: x[1], reverse=True)[:5]
        trait_examples[label] = sbert_cumle_sec([text for text, _ in candidates], max_count=3)
    
    return {
        "predictions": final_preds,
        "binary_predictions": binary_preds,
        "top_traits": top_traits,
        "bottom_traits": bottom_traits,
        "all_traits": differentiated_preds,
        "trait_examples": trait_examples,
        "analyzed_sentences": len(cumleler)
    }

def temizle_cumleler_gelismis(cumleler, full_text):
    filtre_kelimeler = [
        "gmail", "adres", "telefon", "numarası", "istanbul", "ankara",
        "ehliyet", "cinsiyet", "doğum", "türkiye", "mail", "email",
        "hakkımda", "iletişim", "cv", "tc", "nüfus", "uyruk", 
        "tel", "cep", "gsm", "@", "www.", "http"
        "instagram", "linkedin", "twitter", "facebook",
        "sosyal medya", "web sitesi", "blog", "cv", "cv'ye",
        
    ]
    
    temel_bilgiler = bilgi_cikar(full_text)
    
    temiz = []
    for c in cumleler:
        c = c.strip()
        if len(c.split()) < 3:
            continue
        if any(bilgi in c for bilgi in temel_bilgiler):
            continue
        if not any(kw in c.lower() for kw in filtre_kelimeler) and not re.search(r"\d{3,}", c):
            temiz.append(c)
    
    return temiz

def bilgi_cikar(text):
    patterns = [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',#mail
        r'\b\d{3}[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}\b',#tel
        r'\b\d{11}\b',#kimlik
    ]  
    basic_info = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        basic_info.extend(matches)   
    return basic_info

def chunk_text(text, max_length=512):
    words = text.split()
    chunks = []
    current_chunk = []
    
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def sbert_cumle_sec(examples, max_count=3, similarity_threshold=0.7):
    if len(examples) <= max_count:
        return examples   
    if not examples:
        return []  
    selected = [examples[0]]
 
    try:
        
        embeddings = sbert_model.encode(examples, convert_to_tensor=True)
        similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)
        similarity_matrix = similarity_matrix.cpu().numpy()
        #farklı cumle
        for i in range(1, len(examples)):
            if len(selected) >= max_count:
                break
            #benzerlik kontrolu
            is_similar = False
            for j, _ in enumerate(selected):
                selected_idx = examples.index(selected[j])
                if similarity_matrix[i, selected_idx] > similarity_threshold:
                    is_similar = True
                    break
                
            if not is_similar:
                selected.append(examples[i])
    except Exception as e:
        print(f"SBERT çeşitlilik seçiminde hata: {e}")
        #basit
        return examples[:max_count]
    
    return selected


def keyword_finetuning(text):
    text = text.lower()
    keywords = {
        "deneyimli": ["deneyim", "tecrübe", "yönetim", "proje yönetimi", "yazılım geliştirme","rol aldım","yaptım","erasmus+","gördüm",
                    "uzmanlık", "bilgi", "beceri", "şirket", "çalıştım","teknofest","yazdım","erasmus","avrupa","tübitak","kazandım","görev alma","geliştiriyorum","eğitim görüyorum"],
        "profesyonel": ["profesyonel", "detaylı", "disiplinli", "standart", "metodoloji", 
                       "süreç", "prensip", "kalite", "doğruluk", "mükemmeliyet", "sistematik","yönetim kurulu","başkanlık","başkan yardımcılığı","başkan vekilliği","yönetim kurulu üyeliği"],
        "lider": ["lider", "yönetim", "yönlendirme", "ekip yönetimi", "koordinasyon", 
                 "organizasyon", "strateji", "vizyon", "motivasyon", "inisiyatif"],
        "özgüvenli": ["özgüven", "kararlı", "cesaret", "atak", "girişken", "kendine güvenen", 
                     "iddialı", "güçlü", "sağlam", "başarı odaklı","başarılı görüyorum"],
        "abartılı": ["mükemmel", "üstün", "çok iyi", "en iyi", "olağanüstü", "müthiş", 
                    "harika", "uzman", "başarılı", "sıra dışı", "inanılmaz","olmaz","eminim"],
        "takım oyuncusu": ["takım", "işbirliği", "uyum", "beraber", "ekip çalışması","zorluk çekmiyorum","etkinlik","seminer",
                          "paylaşım", "destek", "yardım", "ortak çalışma", "empati", "külüp", "topluluk","geliştirdik","birlikte","teknofest","ettik","bulunduk","gerçekleştirdik"],
        "kararsız": ["bazen", "belki", "olabilir", "tereddüt", "kararsız", "şüphe", 
                    "emin değilim", "değişebilir", "genellikle", "zaman zaman","arada","ara sıra"," bazen","biraz","birazcık","bazen"],
        "içe kapanık": ["sessiz", "sakin", "dinleme", "analiz", "düşünme", "gözlem", 
                       "detay", "yalnız çalışma", "derin düşünce", "içsel","bireysel"],
    }
    
    #skor
    scores = []
    for label in LABELS:
        if label in keywords:
            kw_count = sum([1 for kw in keywords[label] if kw in text])
            score = min(1.0, kw_count / len(keywords[label]) * 2)
            scores.append(score)
        else:
            scores.append(0.5)
    
    return scores

def diff_skor(sorted_preds):
    if not sorted_preds:
        return []
    
    total_traits = len(sorted_preds)
    #diffscore
    diff_factor = 1.3
    
    result = []
    for i, (label, score) in enumerate(sorted_preds):
        position_ratio = (total_traits - i) / total_traits
        new_score = min(1.0, score * (position_ratio * diff_factor))
        result.append((label, new_score))
    return sorted(result, key=lambda x: x[1], reverse=True)

def analiz_calistir(pdf_file, aday_adi, job_desc=""):
    full_text = kelime_ayıkla(pdf_file)
    analysis_results = analiz_cv_kelime(full_text, job_desc)
    kriter_score, matched_kriterler, unmatched_kriterler = degerlendir_kriter(full_text)
    sbert_score = sbert_benzerlik(full_text, job_desc) if job_desc else 0
    
    kriter_weight = 0.6
    sbert_weight = 0.4
    final_score = round((kriter_score * kriter_weight + sbert_score * sbert_weight), 2) if job_desc else kriter_score
    
    return {
        "full_text": full_text,
        "analysis": analysis_results,
        "kriter_score": kriter_score,
        "sbert_score": sbert_score,
        "final_score": final_score,
        "matched_kriterler": matched_kriterler,
        "unmatched_kriterler": unmatched_kriterler
    }


#menü
menu = st.sidebar.selectbox("📌 Menü", ["📥 CV Analiz", "🛠️ Yönetici Alanı"])


    
#frontcvanaliz
if menu == "📥 CV Analiz":
    st.header("📤 CV Yükle ve Analiz Et")

    with st.form("cv_form"):
        uploaded_files = st.file_uploader(
        "📄 Bir veya Birden Fazla PDF CV Yükle", 
        type=["pdf"], 
        accept_multiple_files=True
    )

    # YENİ İŞ İLANI SEÇME
        job_list = get_all_job_descriptions()
        job_options = {job['ilan_adi']: job['ilan_icerik'] for job in job_list}

        selected_job = st.selectbox("💼 İş İlanı Seçin", [""] + list(job_options.keys()))
        job_desc = job_options.get(selected_job, "")  # seçilen iş ilanı içeriğini kullan

        submitted = st.form_submit_button("🔍 Analiz Et", type="primary")

    if submitted and uploaded_files:
        for pdf_file in uploaded_files:
            aday_adi = pdf_file.name.replace(".pdf", "")

            with st.spinner(f"📄 {aday_adi} analiz ediliyor..."):
                analysis = analiz_calistir(pdf_file, aday_adi, job_desc)

                if "error" in analysis.get("analysis", {}):
                    st.error(f"{aday_adi}: {analysis['analysis']['error']}")
                else:
                    st.subheader(f"📈 {aday_adi} Skorları")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("🎯 Kriter Skoru", f"{analysis['kriter_score']:.1f}%")
                    col2.metric("🤝 İş İlanı Benzerliği", f"{analysis['sbert_score']:.1f}%" if job_desc else "N/A")
                    col3.metric("📊 Genel Skor", f"{analysis['final_score']:.1f}%")

                    st.subheader("🧠 Kişilik Özellikleri Analizi")
                    trait_scores = {trait: score for trait, score in analysis["analysis"]["all_traits"]}
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**En Güçlü Özellikler:**")
                        for trait in analysis["analysis"]["top_traits"]:
                            st.success(trait)
                    with col2:
                        st.write("**En Zayıf Özellikler:**")
                        for trait in analysis["analysis"]["bottom_traits"]:
                            st.info(trait)

                    with st.expander("📊 Tüm Özellik Skorları"):
                        for trait, score in analysis["analysis"]["all_traits"]:
                            st.write(f"**{trait}:** {score:.2f}")

                    with st.expander("🔍 Özellik Gösteren Örnek Cümleler"):
                        for trait, examples in analysis["analysis"]["trait_examples"].items():
                            if examples:
                                st.write(f"**{trait.capitalize()} özelliğini gösteren cümleler:**")
                                for ex in examples[:3]:
                                    st.markdown(f"- *\"{ex}\"*")
                                st.write("")

                    with st.expander("📌 Kriter Eşleşme Durumu"):
                        st.markdown("### ✅ Eşleşen Kriterler")
                        if analysis["matched_kriterler"]:
                            for k in analysis["matched_kriterler"]:
                                st.markdown(f"- ✅ **{k['ad']}** (`{k['parametre']}`) - Ağırlık: {k['agirlik']}")
                        else:
                            st.info("Hiçbir kriterle eşleşme bulunamadı.")

                        st.markdown("---")
                        st.markdown("### ❌ Eşleşmeyen Kriterler")
                        if analysis["unmatched_kriterler"]:
                            for k in analysis["unmatched_kriterler"]:
                                st.markdown(f"- ❌ **{k['ad']}** (`{k['parametre']}`) - Ağırlık: {k['agirlik']}")
                        else:
                            st.success("Tüm kriterler eşleşti.")

                    st.caption(f"Toplam {analysis['analysis']['analyzed_sentences']} cümle analiz edildi.")

                    if save_cv(aday_adi, analysis["full_text"], analysis["final_score"]):
                        st.success(f"✅ {aday_adi} veritabanına kaydedildi.")
                        row = {
                            "dosya": pdf_file.name,
                            "cümle": " ".join(temizle_cumleler(extract_meaningful_sentences(analysis["full_text"])))
                        }
                        for label, val in zip(LABELS, analysis["analysis"]["predictions"]):
                            row[label] = round(val, 2)
                        row["is_trained"] = 0

                        df_new = pd.DataFrame([row])
                        try:
                            df_new.to_csv(
                                "toplu_cv_analiz.csv",
                                mode="a",
                                index=False,
                                header=not os.path.exists("toplu_cv_analiz.csv"),
                                quoting=csv.QUOTE_NONNUMERIC,
                                escapechar="\\"
                            )
                            st.success(f"✅ {aday_adi} verisi CSV'ye kaydedildi.")
                        except Exception as e:
                            st.error(f"❌ {aday_adi} CSV kaydında hata oluştu: {e}")
                    else:
                        st.error(f"❌ {aday_adi} kaydedilemedi.")


#frontyönetici
elif menu == "🛠️ Yönetici Alanı":
    if not kimlik_dogrula():
        st.stop()  
    st.subheader("⚙️ Kriter Yönetimi")
    tab1, tab2 = st.tabs(["➕ Ekle", "📋 Listele"])
    with tab1:
        with st.form("kriter_form"):
            ad = st.text_input("Ad")
            tur = st.selectbox("Tür", ["Yetkinlik", "Eğitim", "Dil", "Deneyim","Askerlik"])
            agirlik = st.slider("Ağırlık", 1, 100, 10)
            param = st.text_input("Parametre (örn. Python, B2, Yazılım Geliştirme)")
            if st.form_submit_button("Kaydet"):
                if check_kriter_exists(param):
                    st.warning("⚠️ Bu parametre zaten kayıtlı.")
                elif add_kriter(ad, tur, agirlik, param):
                    st.success("✅ Kriter başarıyla eklendi.")
                    st.rerun() 
    with tab2:
        
        kriterler = get_all_kriterler()
        if not kriterler:
            st.info("Henüz kriter eklenmemiş.")
        else:
            for k in kriterler:
                col1, col2, col3, col4, col5 = st.columns([3, 2, 1, 1, 1])
                col1.write(f"**{k['ad']}**")
                col2.code(k["parametre"])
                col3.metric("Ağırlık", k["agirlik"])

                aktiflik_durumu = "✅ Aktif" if k.get("aktif", True) else "⛔ Pasif"
                if col4.button(aktiflik_durumu, key=f"toggle_{k['id']}"):
                    toggle_kriter_aktiflik(k["id"], not k["aktif"])
                    st.rerun()

                if col5.button("Sil", key=f"sil_{k['id']}"):
                    delete_kriter(k["id"])
                    st.rerun()
            
    st.subheader("📁 Geçmiş Analizler")
    cvs = get_all_cvs()
    if st.button("🗑️ Tüm Kayıtları Sil", type="secondary"):
        delete_all_cvler()
        st.success("✅ Tüm CV kayıtları silindi.")
        st.rerun()
    if not cvs:
        st.info("Henüz analiz edilmiş CV bulunmuyor.")
    else:
        for i, cv in enumerate(cvs):
            # UUID ve skor bilgisini al
            cv_id = cv.get("id") or str(uuid.uuid4())
            
            # Score değerini doğru şekilde al ve formatlama yap
            # Eğer score float değilse veya None ise, 0.0 kabul et
            try:
                cv_score = float(cv.get("puan", 0))  # <- BURASI DÜZELTİLDİ
            except (TypeError, ValueError):
                cv_score = 0.0
                
            cv_name = cv.get("aday_adi", "CV")
            
            # Skorun doğru formatlandığından emin ol
            score_display = f"{cv_score:.2f}%"
            
            # Debug: Puanı kontrol et
            # st.write(f"Debug - Raw score: {cv.get('score')}, Converted: {cv_score}")
            
            # Benzersiz başlık oluştur (ID ve skor bilgisiyle)
            expander_title = f"{i+1}-{cv_name} - {score_display}"
            
            # Her CV için unique ID'li expander kullan
            with st.expander(expander_title):
                # İçerik ve silme butonu için benzersiz anahtarlar
                unique_text_key = f"txt_{cv_id}_{i}"
                unique_button_key = f"del_{cv_id}_{i}"
                
                # CV içeriğini göster
                st.text_area("İçerik", cv["cv_text"], height=200, key=unique_text_key)
                
                # Silme butonu
                if st.button("Sil", key=unique_button_key):
                    delete_cv(cv["id"])
                    st.rerun()
                    
                    

                        
                    
    st.subheader("📝 İş İlanı Yönetimi")
    tab3, tab4 = st.tabs(["➕  İş İlanı Ekle", "📋 İş İlanı Listele"])

    with tab3:
        with st.form("job_form"):
            ilan_adi = st.text_input("İş İlanı Adı")
            ilan_icerik = st.text_area("İş İlanı İçeriği", height=200)
            if st.form_submit_button("Kaydet"):
                if add_job_description(ilan_adi, ilan_icerik):
                    st.success("✅ İş İlanı başarıyla eklendi.")
                    st.rerun()
                else:
                    st.error("❌ İş İlanı eklenirken hata oluştu.")

    with tab4:
        job_list = get_all_job_descriptions()
        if not job_list:
            st.info("Henüz iş ilanı eklenmemiş.")
        else:
            for job in job_list:
                with st.expander(f"{job['ilan_adi']}"):
                    st.text_area("İş İlanı İçeriği", job['ilan_icerik'], height=150, disabled=True)
                    if st.button("Sil", key=f"del_job_{job['id']}"):
                        delete_job_description(job['id'])
                        st.rerun()
        
#eğitimbölümü
    st.subheader("🔁 Model Eğitimi")
    
    try:
        df = pd.read_csv("toplu_cv_analiz.csv", encoding="utf-8")
        untrained_count = len(df[df["is_trained"] != 1]) if "is_trained" in df.columns else len(df)
        st.info(f"Toplam {len(df)} CV verisi mevcut, bunlardan {untrained_count} tanesi henüz eğitilmemiş.")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        st.warning("Henüz eğitim verisi bulunmuyor. CV analiz ederek veri toplayın.")
        untrained_count = 0
    

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📚 Modeli Eğit", disabled=untrained_count==0):
            with st.spinner("Model eğitiliyor... Bu işlem birkaç dakika sürebilir."):
                try:
                    result = os.system("python fine_tune_cv_classifier.py")
                    if result == 0:
                        st.success("✅ Model eğitimi başarıyla tamamlandı.")
                        # Eğitimsonrası
                        try:
                            THRESHOLDS.clear()
                            with open("cv_classifier_model/thresholds.txt", encoding="utf-8") as f:
                                for line in f:
                                    if ":" in line:
                                        label, value = line.strip().split(":", 1)
                                        THRESHOLDS[label] = float(value)
                            st.info("✅ Eşik değerleri güncellendi.")
                        except FileNotFoundError:
                            st.warning("Eşik değerleri dosyası bulunamadı.")
                    else:
                        st.error("❌ Model eğitim sırasında bir hata oluştu.")
                except Exception as e:
                    st.error(f"❌ Hata: {e}")
                    
    with col2:
        if st.button("🗑️ Model Verilerini Temizle"):
            try:
                df = pd.read_csv("toplu_cv_analiz.csv", encoding="utf-8")
                empty_df = pd.DataFrame(columns=df.columns)
                empty_df.to_csv("toplu_cv_analiz.csv", index=False, encoding="utf-8")
                st.success("✅ CSV verileri başarıyla temizlendi.")
                st.rerun()
                
            except FileNotFoundError:
                st.warning("CSV dosyası bulunamadı.")
            except Exception as e:
                
                st.error(f"❌ CSV temizleme sırasında bir hata oluştu: {e}")
         
      
    with st.expander("🧠 Model Bilgileri"):
        st.write("**Mevcut Etiketler:**")
        st.write(", ".join(LABELS))
        