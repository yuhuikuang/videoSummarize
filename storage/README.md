# Storage 模块

## 概述
storage 模块负责向量存储和检索功能，是系统的数据存储层。支持多种向量数据库后端，包括内存存储、Milvus和PostgreSQL+pgvector，提供向量化存储、相似性搜索和RAG（检索增强生成）功能。

## 文件说明

### store.go
向量存储的核心实现文件，包含多种存储后端的实现。

#### 接口定义

**VectorStore**
- `Upsert(jobID, items)`: 插入或更新向量数据
- `Search(jobID, query, topK)`: 向量相似性搜索

#### 存储实现

### 1. MemoryVectorStore（内存存储）
基于内存的向量存储实现，适用于测试和小规模数据。

#### 结构体
- `mu`: 读写锁
- `docs`: 文档映射（jobID -> 文档列表）

#### Document结构
- `Start/End`: 时间范围
- `Text`: 文本内容
- `Summary`: 摘要
- `FramePath`: 帧图片路径
- `Embed`: 词项权重映射

#### 主要方法
- `Upsert(jobID, items)`: 将items转换为Document并存储
- `Search(jobID, query, topK)`: 基于余弦相似度的搜索

### 2. MilvusVectorStore（Milvus向量数据库）
基于Milvus的高性能向量存储实现。

#### 结构体
- `mc`: Milvus客户端
- `coll`: 集合名称
- `dim`: 向量维度
- `oa`: OpenAI客户端

#### 主要方法
- `ensureSchemaAndIndex()`: 确保集合schema和索引
- `embed(text)`: 文本向量化
- `Upsert(jobID, items)`: 向Milvus插入向量数据
- `Search(jobID, query, topK)`: Milvus向量搜索
- `openaiClient()`: 创建OpenAI客户端

### 3. PgVectorStore（PostgreSQL+pgvector）
基于PostgreSQL和pgvector扩展的向量存储实现。

#### 结构体
- `conn`: PostgreSQL连接
- `oa`: OpenAI客户端
- `videoID`: 当前视频ID（数据隔离）

#### 主要方法

**数据管理**
- `SetVideoID(videoID)`: 设置当前视频ID
- `GetVideoID()`: 获取当前视频ID
- `CleanupVideo(videoID)`: 清理指定视频的数据
- `ensureTable()`: 确保数据表存在

**索引管理**
- `createOptimizedVectorIndex()`: 创建优化的向量索引
- `RebuildVectorIndex()`: 重建向量索引
- `GetIndexStatus()`: 获取索引状态
- `AutoRebuildIndexIfNeeded()`: 自动重建索引
- `ScheduleIndexMaintenance()`: 调度索引维护

**向量操作**
- `embed(text)`: 文本向量化
- `Upsert(jobID, items)`: 插入或更新向量数据
- `Search(jobID, query, topK)`: 向量相似性搜索

**客户端管理**
- `openaiClient()`: 创建OpenAI客户端

#### 主要结构体

**StoreRequest**
- `JobID`: 作业ID
- `Items`: 要存储的项目列表

**StoreResponse**
- `JobID`: 作业ID
- `Count`: 存储的项目数量
- `Status`: 状态
- `Message`: 消息

**QueryRequest**
- `JobID`: 作业ID
- `Query`: 查询文本
- `TopK`: 返回结果数量

**QueryResponse**
- `JobID`: 作业ID
- `Query`: 查询文本
- `Hits`: 搜索结果
- `Answer`: RAG生成的答案
- `Status`: 状态
- `Message`: 消息

#### HTTP处理器

**存储接口**
- `StoreHandler()`: 存储HTTP处理器
- `storeHandler()`: 具体的存储处理逻辑

**查询接口**
- `QueryHandler()`: 查询HTTP处理器
- `queryHandler()`: 具体的查询处理逻辑

#### 核心功能函数

**初始化和配置**
- `initVectorStore()`: 初始化向量存储
- `newMilvusVectorStore()`: 创建Milvus存储实例
- `newPgVectorStore()`: 创建PostgreSQL存储实例

**文本处理**
- `tokenize(text)`: 文本分词
- `embedText(text)`: 文本向量化（简单实现）
- `cosine(a, b)`: 计算余弦相似度

**时间格式化**
- `formatTime(seconds)`: 格式化时间显示

**数据操作**
- `storeItems(items, jobID)`: 存储项目数据

**RAG功能**
- `synthesizeAnswer(question, hits)`: 合成答案
- `synthesizeAnswerWithRAG(question, hits)`: 基于RAG的答案生成
- `synthesizeAnswerSimple(question, hits)`: 简单答案合成

**客户端管理**
- `openaiClient()`: 创建OpenAI客户端

## 数据库Schema

### PostgreSQL表结构
```sql
CREATE TABLE IF NOT EXISTS video_segments (
    id SERIAL PRIMARY KEY,
    video_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    text TEXT NOT NULL,
    summary TEXT,
    frame_path TEXT,
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 向量索引
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_video_segments_embedding_cosine 
ON video_segments USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- 查询索引
CREATE INDEX IF NOT EXISTS idx_video_segments_video_job 
ON video_segments(video_id, job_id);

CREATE INDEX IF NOT EXISTS idx_video_segments_time 
ON video_segments(start_time, end_time);
```

### Milvus集合Schema
- **id**: 主键字段
- **job_id**: 作业ID
- **text**: 文本内容
- **summary**: 摘要
- **start_time/end_time**: 时间范围
- **frame_path**: 帧路径
- **embedding**: 向量字段（1536维）

## API接口说明

### 存储接口
- `POST /store`: 存储向量数据
  - 请求体：`StoreRequest`
  - 响应：`StoreResponse`

### 查询接口
- `POST /query`: 向量相似性搜索
  - 请求体：`QueryRequest`
  - 响应：`QueryResponse`

## 特点

1. **多后端支持**: 支持内存、Milvus、PostgreSQL三种存储后端
2. **向量化存储**: 使用OpenAI embeddings进行文本向量化
3. **相似性搜索**: 基于余弦相似度的高效搜索
4. **RAG功能**: 检索增强生成，提供智能问答
5. **索引优化**: 自动索引管理和优化
6. **数据隔离**: 支持多视频数据隔离
7. **并发安全**: 线程安全的操作
8. **自动维护**: 自动索引重建和维护

## 使用方式

### 直接调用
```go
// 初始化存储
err := initVectorStore()
if err != nil {
    log.Fatal(err)
}

// 存储数据
count := globalStore.Upsert(jobID, items)
fmt.Printf("Stored %d items\n", count)

// 搜索
hits := globalStore.Search(jobID, "query text", 5)
for _, hit := range hits {
    fmt.Printf("Score: %.3f, Content: %s\n", hit.Score, hit.Content)
}
```

### HTTP接口调用
```bash
# 存储数据
curl -X POST http://localhost:8080/store \
  -H "Content-Type: application/json" \
  -d '{"job_id": "job123", "items": [...]}'

# 查询数据
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"job_id": "job123", "query": "search text", "top_k": 5}'
```

## 配置说明

### 环境变量
- `VECTOR_STORE_TYPE`: 存储类型（memory/milvus/pgvector）
- `MILVUS_HOST`: Milvus服务器地址
- `MILVUS_PORT`: Milvus端口
- `POSTGRES_URL`: PostgreSQL连接字符串
- `OPENAI_API_KEY`: OpenAI API密钥
- `EMBEDDING_MODEL`: 嵌入模型名称

### 性能调优
- **Milvus**: 调整`nlist`和`nprobe`参数
- **PostgreSQL**: 调整`ivfflat`索引的`lists`参数
- **内存存储**: 适用于小规模数据和测试

## 最佳实践

1. **选择合适的后端**：
   - 小规模/测试：Memory
   - 大规模/生产：Milvus或PostgreSQL
   - 已有PostgreSQL：pgvector

2. **索引优化**：
   - 定期重建索引
   - 监控索引性能
   - 根据数据量调整参数

3. **数据管理**：
   - 及时清理过期数据
   - 使用视频ID进行数据隔离
   - 监控存储空间使用

4. **查询优化**：
   - 合理设置topK值
   - 使用适当的相似度阈值
   - 缓存常用查询结果