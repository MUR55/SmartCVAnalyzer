import sqlite3
import logging
from typing import List, Dict

# loglama
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='cv_degerlendirme.log'
)
logger = logging.getLogger(__name__)

DB_PATH = "cv_degerlendirme.db"

class DatabaseManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance

    def get_connection(self):
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            logger.error(f"❌ SQLite bağlantı hatası: {e}")
            return None

def add_job_description(ilan_adi: str, ilan_icerik: str) -> bool:
    try:
        conn = DatabaseManager().get_connection()
        if not conn: return False
        cursor = conn.cursor()
        cursor.execute("INSERT INTO job_descriptions (ilan_adi, ilan_icerik) VALUES (?, ?)", (ilan_adi, ilan_icerik))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"İş ilanı ekleme hatası: {e}")
        return False
    finally:
        if conn: conn.close()

def get_all_job_descriptions() -> List[Dict]:
    conn = DatabaseManager().get_connection()
    if not conn: return []
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, ilan_adi, ilan_icerik FROM job_descriptions ORDER BY id DESC")
        return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"İş ilanı çekme hatası: {e}")
        return []
    finally:
        conn.close()

def delete_job_description(job_id: int) -> bool:
    try:
        conn = DatabaseManager().get_connection()
        if not conn: return False
        cursor = conn.cursor()
        cursor.execute("DELETE FROM job_descriptions WHERE id = ?", (job_id,))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        logger.error(f"İş ilanı silme hatası: {e}")
        return False
    finally:
        if conn: conn.close()

def get_all_kriterler() -> List[Dict]:
    conn = DatabaseManager().get_connection()
    if not conn: return []
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, ad, tur, agirlik, parametre, aktif, created_at FROM kriterler ORDER BY created_at DESC")
        results = [dict(row) for row in cursor.fetchall()]
        for r in results:
            r["agirlik"] = int(r["agirlik"]) if r["agirlik"] is not None else 0
            r["aktif"] = bool(int(r["aktif"])) if r["aktif"] is not None else True
        return results
    except Exception as e:
        logger.error(f"Kriter alma hatası: {e}")
        return []
    finally:
        conn.close()

def add_kriter(ad: str, tur: str, agirlik: int, parametre: str) -> bool:
    if check_kriter_exists(parametre):
        logger.warning(f"Kriter zaten var: {parametre}")
        return False
    try:
        conn = DatabaseManager().get_connection()
        if not conn: return False
        cursor = conn.cursor()
        cursor.execute("INSERT INTO kriterler (ad, tur, agirlik, parametre) VALUES (?, ?, ?, ?)", (ad, tur, agirlik, parametre))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Kriter ekleme hatası: {e}")
        return False
    finally:
        if conn: conn.close()

def delete_kriter(kriter_id: int) -> bool:
    try:
        conn = DatabaseManager().get_connection()
        if not conn: return False
        cursor = conn.cursor()
        cursor.execute("DELETE FROM kriterler WHERE id = ?", (kriter_id,))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        logger.error(f"Kriter silme hatası: {e}")
        return False
    finally:
        if conn: conn.close()

def check_kriter_exists(parametre: str) -> bool:
    conn = DatabaseManager().get_connection()
    if not conn: return False
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM kriterler WHERE parametre = ?", (parametre,))
        return cursor.fetchone() is not None
    except Exception as e:
        logger.error(f"Kriter kontrol hatası: {e}")
        return False
    finally:
        conn.close()

def save_cv(aday_adi: str, cv_text: str, puan: float) -> bool:
    try:
        conn = DatabaseManager().get_connection()
        if not conn: return False
        cursor = conn.cursor()
        cursor.execute("INSERT INTO cvler (aday_adi, cv_text, puan) VALUES (?, ?, ?)", (aday_adi, cv_text, puan))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"CV kaydetme hatası: {e}")
        return False
    finally:
        if conn: conn.close()

def get_all_cvs() -> List[Dict]:
    conn = DatabaseManager().get_connection()
    if not conn: return []
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, aday_adi, cv_text, puan, tarih FROM cvler ORDER BY tarih DESC")
        return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"CV çekme hatası: {e}")
        return []
    finally:
        conn.close()

def delete_cv(cv_id: int) -> bool:
    try:
        conn = DatabaseManager().get_connection()
        if not conn: return False
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cvler WHERE id = ?", (cv_id,))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        logger.error(f"CV silme hatası: {e}")
        return False
    finally:
        if conn: conn.close()

def delete_all_cvler() -> bool:
    try:
        conn = DatabaseManager().get_connection()
        if not conn: return False
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cvler")
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Tüm CV'leri silme hatası: {e}")
        return False
    finally:
        if conn: conn.close()

def toggle_kriter_aktiflik(kriter_id: int, yeni_durum: bool) -> bool:
    try:
        conn = DatabaseManager().get_connection()
        if not conn: return False
        cursor = conn.cursor()
        cursor.execute("UPDATE kriterler SET aktif = ? WHERE id = ?", (int(yeni_durum), kriter_id))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Kriter aktiflik güncelleme hatası: {e}")
        return False
    finally:
        if conn: conn.close()

def save_analiz_result(aday_adi: str, full_text: str, kriter_skoru: float,
                       sbert_skoru: float, genel_skor: float,
                       top_traits: List[str], bottom_traits: List[str]) -> bool:
    try:
        conn = DatabaseManager().get_connection()
        if not conn:
            return False
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO cv_analizler (
                aday_adi, full_text, kriter_skoru, sbert_skoru, genel_skor, 
                top_traits, bottom_traits
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            aday_adi,
            full_text,
            kriter_skoru,
            sbert_skoru,
            genel_skor,
            ', '.join(top_traits),
            ', '.join(bottom_traits)
        ))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Analiz sonucu kaydetme hatası: {e}")
        return False
    finally:
        if conn: conn.close()
