// AI Automation Bot - MongoDB Initialization Script
// MongoDB veritabanı başlatma scripti

// Veritabanını seç
db = db.getSiblingDB('automation_db');

// Koleksiyonları oluştur
db.createCollection('users');
db.createCollection('tasks');
db.createCollection('task_executions');
db.createCollection('models');
db.createCollection('data_sources');
db.createCollection('reports');
db.createCollection('system_metrics');
db.createCollection('logs');

// İndeksler oluştur
db.users.createIndex({ "username": 1 }, { unique: true });
db.users.createIndex({ "email": 1 }, { unique: true });
db.tasks.createIndex({ "name": 1 }, { unique: true });
db.tasks.createIndex({ "status": 1 });
db.task_executions.createIndex({ "task_id": 1 });
db.task_executions.createIndex({ "started_at": 1 });
db.models.createIndex({ "name": 1, "version": 1 });
db.data_sources.createIndex({ "name": 1 }, { unique: true });
db.reports.createIndex({ "report_type": 1 });
db.system_metrics.createIndex({ "metric_name": 1, "recorded_at": 1 });
db.logs.createIndex({ "timestamp": 1 });
db.logs.createIndex({ "level": 1 });

// Varsayılan admin kullanıcısı oluştur
db.users.insertOne({
    username: "admin",
    email: "admin@example.com",
    password_hash: "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj4J/HS.iK8i",
    is_active: true,
    is_admin: true,
    created_at: new Date(),
    updated_at: new Date()
});

// Varsayılan görevler oluştur
db.tasks.insertMany([
    {
        name: "data_collection",
        description: "Veri toplama görevi",
        schedule_expr: "0 */6 * * *",
        is_enabled: true,
        status: "enabled",
        created_at: new Date(),
        updated_at: new Date()
    },
    {
        name: "model_training",
        description: "Model eğitimi görevi",
        schedule_expr: "0 2 * * *",
        is_enabled: true,
        status: "enabled",
        created_at: new Date(),
        updated_at: new Date()
    },
    {
        name: "report_generation",
        description: "Rapor oluşturma görevi",
        schedule_expr: "0 9 * * *",
        is_enabled: true,
        status: "enabled",
        created_at: new Date(),
        updated_at: new Date()
    }
]);

// Varsayılan veri kaynakları oluştur
db.data_sources.insertMany([
    {
        name: "example_api",
        source_type: "api",
        config: {
            url: "https://api.example.com/data",
            method: "GET",
            headers: {}
        },
        is_enabled: true,
        created_at: new Date(),
        updated_at: new Date()
    },
    {
        name: "example_web",
        source_type: "web_scraping",
        config: {
            url: "https://example.com",
            selectors: {
                title: "h1"
            }
        },
        is_enabled: true,
        created_at: new Date(),
        updated_at: new Date()
    }
]);

print("MongoDB initialization completed successfully!"); 