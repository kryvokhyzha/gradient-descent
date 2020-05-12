import sqlite3 as sql 


def db_creation():
    conn = sql.connect('db/user_history.db')
    cur = conn.cursor()

    cur.executescript('''

        CREATE TABLE IF NOT EXISTS AlgorithmParams(
            id INTEGER PRIMARY KEY,
            hypothesis TEXT NOT NULL,
            cost_function TEXT NOT NULL,
            reqularization TEXT NOT NULL,
            scalling_function TEXT NOT NULL,
            reg_coef REAL NOT NULL,
            learning_rate REAL NOT NULL,
            eps REAL NOT NULL,
            max_num_itter INT NOT NULL,
            weights TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS UserHistory(
            id INTEGER PRIMARY KEY,
            algorithm_name TEXT NOT NULL,
            execution_time TEXT NOT NULL,
            ins_date timestamp NOT NULL,
            param_id INT NOT NULL,
            FOREIGN KEY (param_id)
                REFERENCES AlgorithmParams(param_id)
        );

    ''')

    conn.commit()
    conn.close()

def db_clean():
    conn = sql.connect('db/user_history.db')
    cur = conn.cursor()

    cur.executescript(''' 
            DELETE FROM UserHistory;
            DELETE FROM AlgorithmParams;
    ''')

    conn.commit()
    conn.close()
