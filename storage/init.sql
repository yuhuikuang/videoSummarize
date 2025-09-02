-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create videos table for video metadata
CREATE TABLE IF NOT EXISTS videos (
    id SERIAL PRIMARY KEY,
    video_id VARCHAR(255) UNIQUE NOT NULL,
    video_name VARCHAR(500),
    video_path VARCHAR(1000),
    duration FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create video_segments table for storing video segments with embeddings
CREATE TABLE IF NOT EXISTS video_segments (
    id SERIAL PRIMARY KEY,
    video_id VARCHAR(255) NOT NULL,
    job_id VARCHAR(255) NOT NULL,
    segment_id VARCHAR(255) NOT NULL,
    start_time FLOAT NOT NULL,
    end_time FLOAT NOT NULL,
    text TEXT NOT NULL,
    summary TEXT,
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(video_id, job_id, segment_id),
    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
);

-- Create indexes for better performance and isolation
CREATE INDEX IF NOT EXISTS idx_video_segments_video_id ON video_segments(video_id);
CREATE INDEX IF NOT EXISTS idx_video_segments_job_id ON video_segments(job_id);
CREATE INDEX IF NOT EXISTS idx_video_segments_video_job ON video_segments(video_id, job_id);

-- Create partitioned index on embedding for vector similarity search by video_id
CREATE INDEX IF NOT EXISTS idx_video_segments_embedding ON video_segments 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Create composite index for efficient video-specific searches
CREATE INDEX IF NOT EXISTS idx_video_segments_video_embedding ON video_segments(video_id) 
INCLUDE (embedding);

-- Add function to clean up segments for a specific video
CREATE OR REPLACE FUNCTION cleanup_video_segments(target_video_id VARCHAR)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM video_segments WHERE video_id = target_video_id;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    DELETE FROM videos WHERE video_id = target_video_id;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;