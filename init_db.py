import sqlite3

conn = sqlite3.connect('turnike.db')
cursor = conn.cursor()

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
