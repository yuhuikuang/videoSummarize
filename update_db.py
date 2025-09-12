import psycopg2

# 连接数据库
conn = psycopg2.connect('postgres://postgres:postgres@localhost:5432/videosummarize?sslmode=disable')
cur = conn.cursor()

print('=== 更新数据库结构 ===')

try:
    # 添加embedding列到video_segments表
    print('添加embedding列到video_segments表...')
    cur.execute("ALTER TABLE video_segments ADD COLUMN IF NOT EXISTS embedding vector(1536)")
    
    # 创建embedding索引
    print('创建embedding索引...')
    cur.execute("CREATE INDEX IF NOT EXISTS idx_video_segments_embedding ON video_segments USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)")
    
    # 提交更改
    conn.commit()
    print('✅ 数据库结构更新成功！')
    
    # 验证更新
    print('\n=== 验证更新后的表结构 ===')
    cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'video_segments' ORDER BY ordinal_position")
    for row in cur.fetchall():
        print(f'{row[0]}: {row[1]}')
        
except Exception as e:
    print(f'❌ 更新失败: {e}')
    conn.rollback()

conn.close()