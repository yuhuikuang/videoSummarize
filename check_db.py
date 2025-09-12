import psycopg2
import json

# 连接数据库
conn = psycopg2.connect(
    host="localhost",
    database="videosummarize",
    user="postgres",
    password="postgres"
)
cur = conn.cursor()

# 首先找到最新的job_id
print("=== 查找最新任务 ===")
cur.execute("""
    SELECT job_id, COUNT(*) as segment_count, MAX(created_at) as latest_time
    FROM video_segments 
    GROUP BY job_id 
    ORDER BY MAX(created_at) DESC 
    LIMIT 5
""")
latest_jobs = cur.fetchall()
for i, row in enumerate(latest_jobs):
    print(f"{i+1}. Job ID: {row[0]}, Segments: {row[1]}, Latest: {row[2]}")

# 检查特定的任务ID
specific_job_id = "87d880766d2759646dbe5b2aa5701b29"
print(f"\n=== 检查特定任务 {specific_job_id} ===")
cur.execute("SELECT COUNT(*) FROM video_segments WHERE job_id = %s", (specific_job_id,))
total_segments = cur.fetchone()[0]
print(f"Total segments for job {specific_job_id}: {total_segments}")

if total_segments > 0:
    cur.execute("SELECT COUNT(*) FROM video_segments WHERE job_id = %s AND embedding IS NOT NULL", (specific_job_id,))
    embedding_segments = cur.fetchone()[0]
    print(f"Segments with embeddings: {embedding_segments}")
    
    # 检查该job_id的记录详情
    print(f"\n=== 任务 {specific_job_id} 的记录详情 ===")
    cur.execute("SELECT id, LEFT(text, 100) as text_preview, embedding IS NOT NULL as has_embedding FROM video_segments WHERE job_id = %s LIMIT 5", (specific_job_id,))
    for row in cur.fetchall():
        print(f"ID: {row[0]}, Text: {row[1]}, Has Embedding: {row[2]}")
else:
    print(f"任务 {specific_job_id} 没有找到数据")

# 检查所有数据的embedding情况
print("\n=== 检查所有数据的embedding情况 ===")
cur.execute("SELECT COUNT(*) FROM video_segments")
total_all = cur.fetchone()[0]
print(f"Total segments in database: {total_all}")

cur.execute("SELECT COUNT(*) FROM video_segments WHERE embedding IS NOT NULL")
embedding_all = cur.fetchone()[0]
print(f"Segments with embeddings: {embedding_all}")
print(f"Embedding coverage: {embedding_all/total_all*100:.1f}%" if total_all > 0 else "No data")

# 检查ASR配置
print("\n=== 检查ASR提供商配置 ===")
with open('config.json', 'r') as f:
    config = json.load(f)
    print(f"ASR Provider: {config.get('asr_provider', 'Not set')}")
    print(f"API Key configured: {'api_key' in config and config['api_key'] != ''}")

conn.close()
print("\n检查完成！")