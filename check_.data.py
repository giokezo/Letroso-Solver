import sqlite3

def display_results():
    conn = sqlite3.connect('letroso_results.db')
    cursor = conn.cursor()

    query = """
        SELECT 
            g.session_id, 
            s.student_name,
            s.login_time,
            g.opener,  
            g.score 
        FROM algorithm_scores g
        JOIN sessions s ON g.session_id = s.session_id
        ORDER BY g.game_id ASC
    """

    print(f"{'ID':<4} | {'TIME':<19} | {'NAME':<12} | {'OPENER':<12} | {'SCORE'}")
    print("-" * 70)
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        
        if not rows:
            print("No data found in database.")
        
        for row in rows:

             print(f"{row[0]:<4} | {row[2]:<19} | {row[1]:<12} | {row[3]:<12} | {row[4]}")
            
    except sqlite3.OperationalError as e:
        print(f"Error: {e}. Make sure you ran database.py first.")
    
    conn.close()

if __name__ == "__main__":
    display_results()