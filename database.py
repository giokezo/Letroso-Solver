import sqlite3

def init_db():
    conn = sqlite3.connect('letroso_results.db')
    cursor = conn.cursor()
    

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT,
            login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS algorithm_scores (
            game_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            opener TEXT,
            score INTEGER,
            FOREIGN KEY (session_id) REFERENCES sessions (session_id)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS manual_scores (
            manual_id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name TEXT,
            score INTEGER
        )
    ''')
    conn.commit()
    conn.close()
    print("Database tables initialized.")
    

def start_new_session(name):
    conn = sqlite3.connect('letroso_results.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO sessions (student_name) VALUES (?)', (name,))
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id

def save_game_score(session_id, opener, score):
    conn = sqlite3.connect('letroso_results.db')
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO algorithm_scores (session_id, opener, score) VALUES (?, ?, ?)', 
        (session_id, opener, score)
    )
    conn.commit()
    conn.close()

def add_manual_scores_list(name, scores_list):
    """Saves a name and a list of scores into the manual table."""
    conn = sqlite3.connect('letroso_results.db')
    cursor = conn.cursor()
    for s in scores_list:
        cursor.execute('INSERT INTO manual_scores (player_name, score) VALUES (?, ?)', (name, s))
    conn.commit()
    conn.close()
    print(f"Added {len(scores_list)} manual scores for {name}.")

def clear_manual_scores():
    """DELETES EVERYTHING from the manual_scores table."""
    conn = sqlite3.connect('letroso_results.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM manual_scores')
    conn.commit()
    conn.close()
    print("All manual scores have been cleared.")

def delete_manual_by_name(name):
    """DELETES only entries for a specific person (e.g., 'Mariami')."""
    conn = sqlite3.connect('letroso_results.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM manual_scores WHERE player_name = ?', (name,))
    conn.commit()
    conn.close()
    print(f"Deleted all manual entries for: {name}")

if __name__ == "__main__":
    init_db()

    mariami_scores = [5, 5, 8, 3, 6, 16, 9, 10, 13, 7, 30, 7, 5, 2, 2, 4, 19, 8, 14, 13]
    add_manual_scores_list("Mariami", mariami_scores)

    daviti_scores = [15, 11, 6, 11, 17, 8, 10, 8, 5, 13, 10, 21, 17, 11, 12, 16, 8 , 8, 11, 16]
    add_manual_scores_list("Daviti", daviti_scores)

    semi_scores = [5, 2, 29, 25, 9, 16, 9, 3, 11, 14, 13, 13, 8, 4, 10, 8, 12, 10, 9, 7]
    add_manual_scores_list("Semi", semi_scores)

    tatia_scores = [6, 2, 9, 4, 7, 8, 8, 12, 7, 6, 5, 5, 4, 7, 15, 4, 10, 2, 4, 6]
    add_manual_scores_list("Tatia", tatia_scores)

    tako_scores = [4, 6, 3, 7, 3, 3, 8, 5, 6, 2, 3, 4, 7, 5, 4, 4, 5, 4, 4, 6]
    add_manual_scores_list("Tako", tako_scores)

    eka_scores = [2, 3, 2, 3, 3, 4, 2, 3, 4, 3, 2, 4, 2, 4, 2, 3, 5, 3, 2, 4, 2, 4, 3]
    add_manual_scores_list("Eka", eka_scores)


    nikolozi_scores = [30, 7, 20, 15, 34, 6, 33, 19, 10, 21, 30, 29, 8, 23, 7, 5, 26, 14, 10, 13]
    add_manual_scores_list("nikolozi", nikolozi_scores)

    ana_scores = [6, 5, 35, 34, 3, 32, 12, 11, 30, 10, 29, 28, 7, 27, 26, 5, 5, 14, 23, 13]
    add_manual_scores_list("ana", ana_scores)

    giorgi_scores = [19, 8, 37, 6, 15, 5, 11, 3, 12, 21, 11, 20, 9, 28, 5, 4, 26, 5, 4, 24]
    add_manual_scores_list("giorgi", giorgi_scores)

    
    tako1_scores = [25, 5, 4, 4, 23, 2, 11, 21, 6, 3, 3, 4, 16, 2, 14, 3, 21, 7, 8, 10]
    add_manual_scores_list("tako", tako1_scores)

    wula_scores = [6, 26, 5, 35, 6, 14, 21, 4, 21, 12, 11, 9, 13, 8, 5, 11, 9, 8, 9, 9]
    add_manual_scores_list("daviti", wula_scores)

    elene_scores = [4, 6, 14, 6, 4, 21, 4, 12, 11, 10, 6, 10, 4, 9, 12, 9, 19, 28, 4, 7]
    add_manual_scores_list("elene", elene_scores)

    saba_scores = [7, 6, 3, 6, 6, 5, 7, 5, 5, 5, 19, 4, 4, 3, 4, 3, 15, 4, 4, 8]
    add_manual_scores_list("saba", saba_scores)

    mariami1_scores = [11, 23, 31, 6, 7, 15, 4, 14, 13, 21, 8, 8, 8, 3, 14, 32, 29, 9, 4, 4]
    add_manual_scores_list("mariami", mariami1_scores)
    #clear_manual_scores()
        




