import sqlite3
import csv

# ====== CONFIG ======
db_file = "drugs.db"     # <-- put your .db filename here
table_name = "drugs"    # <-- put your table name here
field_name = "FirstName"     # <-- the field we want
output_csv = "output.csv"
keywords = ["TAB", "TABS","TAB.", "TABS.", "AMP", "AMPS","AMP.", "AMPS.", "SACHET", "SACHETS", "SACHET.", "SACHETS.", "CAPSULE", "CAPSULES", "CAPSULE.", "CAPSULES."]

# ====== CONNECT TO DB ======
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# ====== QUERY DATA ======
cursor.execute(f"SELECT {field_name} FROM {table_name}")
rows = cursor.fetchall()

processed_rows = []

for row in rows:
    if row[0] is None:
        continue
    words = row[0].strip().split()
    if not words:
        continue

    # take first 3 words
    first_three = words[:3]

    # check last word for keywords
    last_word = words[-1].upper()
    if last_word in keywords:
        # add last word if not already in first three
        if last_word not in [w.upper() for w in first_three]:
            first_three.append(words[-1])

    processed_rows.append(" ".join(first_three))

# ====== SAVE TO CSV ======
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([field_name])  # header
    for item in processed_rows:
        writer.writerow([item])

print(f"Processed {len(processed_rows)} rows. Saved to {output_csv}.")
