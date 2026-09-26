import psycopg2
conn = psycopg2.connect("postgres://tsdbadmin:Londonbridge1%21@c17gvz9mat.iqazmqohrx.tsdb.cloud.timescale.com:31699/tsdb?sslmode=require")
cur = conn.cursor()
cur.execute("""
    SELECT reading_time, COUNT(*) 
    FROM meter_readings 
    WHERE meter_id = '0179002097105' 
    GROUP BY reading_time 
    HAVING COUNT(*) > 1
    ORDER BY reading_time
    LIMIT 10
""")
rows = cur.fetchall()
print(f"Duplicate reading_time entries found: {len(rows)}")
for r in rows:
    print(r)