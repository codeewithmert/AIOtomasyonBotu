-- AI Automation Bot - Database Initialization Script
-- PostgreSQL veritabanı başlatma scripti

-- Veritabanını oluştur (eğer yoksa)
-- CREATE DATABASE automation_db;

-- Kullanıcı oluştur (eğer yoksa)
-- CREATE USER ai_bot_user WITH PASSWORD 'your_secure_password';

-- Yetkileri ver
GRANT ALL PRIVILEGES ON DATABASE automation_db TO ai_bot_user;

-- Veritabanına bağlan
\c automation_db;

-- Temel tabloları oluştur
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tasks (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    schedule_expr VARCHAR(100),
    is_enabled BOOLEAN DEFAULT TRUE,
    last_run TIMESTAMP,
    next_run TIMESTAMP,
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS task_executions (
    id SERIAL PRIMARY KEY,
    task_id INTEGER REFERENCES tasks(id),
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    status VARCHAR(20) DEFAULT 'running',
    error_message TEXT,
    execution_time_ms INTEGER
);

CREATE TABLE IF NOT EXISTS models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    file_path VARCHAR(255) NOT NULL,
    accuracy FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS data_sources (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    source_type VARCHAR(50) NOT NULL, -- 'api', 'web_scraping', 'database', 'file'
    config JSONB NOT NULL,
    is_enabled BOOLEAN DEFAULT TRUE,
    last_collection TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS reports (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    report_type VARCHAR(50) NOT NULL,
    file_path VARCHAR(255),
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'generated',
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS system_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(50) NOT NULL,
    metric_value FLOAT NOT NULL,
    unit VARCHAR(20),
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- İndeksler oluştur
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_task_executions_task_id ON task_executions(task_id);
CREATE INDEX IF NOT EXISTS idx_task_executions_started_at ON task_executions(started_at);
CREATE INDEX IF NOT EXISTS idx_models_name_version ON models(name, version);
CREATE INDEX IF NOT EXISTS idx_data_sources_type ON data_sources(source_type);
CREATE INDEX IF NOT EXISTS idx_reports_type ON reports(report_type);
CREATE INDEX IF NOT EXISTS idx_system_metrics_name_time ON system_metrics(metric_name, recorded_at);

-- Varsayılan admin kullanıcısı oluştur (şifre: admin123)
INSERT INTO users (username, email, password_hash, is_admin) 
VALUES ('admin', 'admin@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj4J/HS.iK8i', TRUE)
ON CONFLICT (username) DO NOTHING;

-- Varsayılan görevler oluştur
INSERT INTO tasks (name, description, schedule_expr, status) VALUES
('data_collection', 'Veri toplama görevi', '0 */6 * * *', 'enabled'),
('model_training', 'Model eğitimi görevi', '0 2 * * *', 'enabled'),
('report_generation', 'Rapor oluşturma görevi', '0 9 * * *', 'enabled')
ON CONFLICT (name) DO NOTHING;

-- Varsayılan veri kaynakları oluştur
INSERT INTO data_sources (name, source_type, config) VALUES
('example_api', 'api', '{"url": "https://api.example.com/data", "method": "GET", "headers": {}}'),
('example_web', 'web_scraping', '{"url": "https://example.com", "selectors": {"title": "h1"}}')
ON CONFLICT (name) DO NOTHING;

-- Commit değişiklikleri
COMMIT; 