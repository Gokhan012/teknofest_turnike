import sqlite3

def setup_database():
    try:
        conn = sqlite3.connect('turnike.db')
        cursor = conn.cursor()

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            student_id TEXT UNIQUE,
            embedding BLOB NOT NULL
        )
        ''')

        conn.commit()
        
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
