import sqlite3

def setup_database():
    try:
        # Veritabanına bağlan (Dosya yoksa oluşur)
        conn = sqlite3.connect('turnike.db')
        cursor = conn.cursor()

        # Tabloyu oluştur (IF NOT EXISTS hata almanı engeller)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            student_id TEXT UNIQUE,
            embedding BLOB NOT NULL
        )
        ''')

        # Değişiklikleri kaydet
        conn.commit()
        
        # Tablonun oluşup oluşmadığını test et
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users';")
        if cursor.fetchone():
            print("✅ 'users' tablosu başarıyla oluşturuldu ve hazır!")
        else:
            print("❌ Tablo oluşturulamadı.")

        conn.close()
    except Exception as e:
        print(f"⚠️ Bir hata oluştu: {e}")

if __name__ == "__main__":
    setup_database()
