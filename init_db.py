import sqlite3

# Yeni ve temiz bir dosya oluşturur
conn = sqlite3.connect('turnike.db')
cursor = conn.cursor()

# Tabloyu oluştur (Vektörleri BLOB olarak saklayacağız)
cursor.execute('''
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    student_id TEXT UNIQUE,
    embedding BLOB NOT NULL
)
''')

conn.commit()
conn.close()
print("Sıfır kilometre 'turnike.db' oluşturuldu!")
