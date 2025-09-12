import psycopg2

# 连接数据库
conn = psycopg2.connect(
    host="localhost",
    database="videosummarize",
    user="postgres",
    password="postgres"
)
cur = conn.cursor()

print("=== 修复embedding维度问题 ===")
print("当前问题：生成的embedding是1024维，但数据库期望1536维")

try:
    # 1. 检查当前embedding列的约束
    print("\n1. 检查当前embedding列约束...")
    cur.execute("""
        SELECT conname, pg_get_constraintdef(oid) 
        FROM pg_constraint 
        WHERE conrelid = 'video_segments'::regclass 
        AND conname LIKE '%embedding%'
    """)
    constraints = cur.fetchall()
    for constraint in constraints:
        print(f"约束: {constraint[0]} - {constraint[1]}")
    
    # 2. 删除现有的embedding维度约束（如果存在）
    print("\n2. 删除现有的embedding维度约束...")
    for constraint in constraints:
        if 'embedding' in constraint[0].lower():
            print(f"删除约束: {constraint[0]}")
            cur.execute(f"ALTER TABLE video_segments DROP CONSTRAINT IF EXISTS {constraint[0]}")
    
    # 3. 检查当前embedding列的数据类型
    print("\n3. 检查embedding列数据类型...")
    cur.execute("""
        SELECT column_name, data_type, character_maximum_length
        FROM information_schema.columns 
        WHERE table_name = 'video_segments' AND column_name = 'embedding'
    """)
    column_info = cur.fetchone()
    if column_info:
        print(f"列信息: {column_info}")
    
    # 4. 重新创建embedding列为1024维
    print("\n4. 重新创建embedding列为1024维...")
    
    # 先删除现有列
    cur.execute("ALTER TABLE video_segments DROP COLUMN IF EXISTS embedding")
    print("✓ 已删除现有embedding列")
    
    # 创建新的1024维embedding列
    cur.execute("ALTER TABLE video_segments ADD COLUMN embedding vector(1024)")
    print("✓ 已创建新的1024维embedding列")
    
    # 5. 创建索引以提高查询性能
    print("\n5. 创建embedding索引...")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_video_segments_embedding ON video_segments USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)")
    print("✓ 已创建embedding索引")
    
    # 6. 验证修复结果
    print("\n6. 验证修复结果...")
    cur.execute("""
        SELECT column_name, data_type, character_maximum_length
        FROM information_schema.columns 
        WHERE table_name = 'video_segments' AND column_name = 'embedding'
    """)
    new_column_info = cur.fetchone()
    if new_column_info:
        print(f"新列信息: {new_column_info}")
    
    # 提交更改
    conn.commit()
    print("\n✓ 修复完成！现在embedding列支持1024维度")
    
except Exception as e:
    print(f"\n❌ 修复过程中出现错误: {e}")
    conn.rollback()
    
finally:
    conn.close()
    print("\n数据库连接已关闭")