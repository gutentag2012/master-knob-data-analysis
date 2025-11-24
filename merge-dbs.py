import sqlite3

db_files = ['knob-data-collection_2.db', 'knob-data-collection_single_3.db', 'knob-data-collection_single_4.db']
merged = sqlite3.connect('merged.sqlite')

# Copy schema from the first DB
with sqlite3.connect(db_files[0]) as src:
    schema_script = ""
    for row in src.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL;"):
        stmt = row[0].strip()
        if not stmt.endswith(";"):
            stmt += ";"
        schema_script += stmt + "\n\n"
    merged.executescript(schema_script)

# Merge all data (UUIDs make this easy)
for db_path in db_files:
    print(f"Merging {db_path}...")
    with sqlite3.connect(db_path) as src:
        src.row_factory = sqlite3.Row
        tables = [
            row[0]
            for row in src.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        for table in tables:
            rows = src.execute(f"SELECT * FROM {table}").fetchall()
            if not rows:
                continue
            cols = [d[1] for d in src.execute(f'PRAGMA table_info({table})')]  # <-- fixed here
            placeholders = ", ".join("?" for _ in cols)
            sql = f"INSERT OR IGNORE INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
            merged.executemany(sql, [[r[c] for c in cols] for r in rows])

merged.commit()
merged.close()
print("✅ Merge complete!")

# Afterwards execute this SQL to clean up duplicates if any
# select s.participant_id, count(*) 
#     from session s left join sessionTask st on s.id = st.session_id
#     group by s.participant_id;

# delete from session where participant_id = '1cc7a246-7caf-43ce-a950-dc51f519edb0';
# delete from sessionTask where session_id = '8461f300-4e70-4041-a7e0-9bf746602119';
# delete from sessionTaskMarker where session_task_id = '2bc28cf8-505d-4618-8d94-8c03584261e4';
# delete from sensorData where session_task_id = '2bc28cf8-505d-4618-8d94-8c03584261e4';
