-- =======================================
-- 1. Tạo Database
-- =======================================
CREATE DATABASE voice_auth CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE voice_auth;

-- =======================================
-- 2. Bảng users (tài khoản người dùng)
-- =======================================
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    email VARCHAR(150) UNIQUE,
    phone VARCHAR(20) UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role ENUM('user','admin') DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- =======================================
-- 3. Bảng voice_samples (mẫu giọng gốc)
-- =======================================
CREATE TABLE voice_samples (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    file_path VARCHAR(255),
    embedding_vector BLOB,  -- có thể thay bằng JSON nếu MySQL 8+
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- =======================================
-- 4. Bảng auth_logs (lịch sử xác thực)
-- =======================================
CREATE TABLE auth_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NULL,
    audio_input_path VARCHAR(255),
    is_real_voice BOOLEAN,
    matched_user_id INT NULL,
    result ENUM('success','fail') NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
    FOREIGN KEY (matched_user_id) REFERENCES users(id) ON DELETE SET NULL
);

-- =======================================
-- 5. Bảng models (danh sách model ML)
-- =======================================
CREATE TABLE models (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    version VARCHAR(50),
    description TEXT,
    status ENUM('active','deprecated','training') DEFAULT 'active',
    file_path VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- =======================================
-- 6. Bảng training_datasets (mẫu training cho user)
-- =======================================
CREATE TABLE training_datasets (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    file_path VARCHAR(255),   -- đường dẫn file audio training
    embedding_vector BLOB,    -- vector đặc trưng (nếu cần)
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- =======================================
-- 7. Bảng model_training_runs (1 lần huấn luyện model)
-- =======================================
CREATE TABLE model_training_runs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    model_id INT NOT NULL,
    version VARCHAR(50),                  -- phiên bản model trong run này
    total_epochs INT,                     -- số epoch dự kiến
    status ENUM('success','fail','running') DEFAULT 'running',
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    finished_at TIMESTAMP NULL,
    FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE
);

-- =======================================
-- 8. Bảng training_logs (log chi tiết theo epoch/step)
-- =======================================
CREATE TABLE training_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    run_id INT NOT NULL,
    epoch INT,
    accuracy FLOAT,
    loss FLOAT,
    log_path VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES model_training_runs(id) ON DELETE CASCADE
);

-- =======================================
-- 9. Dữ liệu mẫu (sample data)
-- =======================================

-- Chèn 1 user
INSERT INTO users (username, email, password_hash, role)
VALUES ('alice', 'alice@example.com', '$2b$10$YwysaxJ33/5t7cYdk9iA/uVLgE6aiYYlbZuk2n.R4ObgSc7miqKeu', 'user');

-- Chèn 1 mẫu giọng cho user
INSERT INTO voice_samples (user_id, file_path)
VALUES (1, '/audio/voice_alice.wav');

-- Chèn 1 model
INSERT INTO models (name, version, description, status, file_path)
VALUES ('SpeakerID-v2', '1.0', 'Model nhận dạng giọng nói', 'active', '/models/speakerid_v2.h5');

-- Chèn mẫu training cho user
INSERT INTO training_datasets (user_id, file_path, description)
VALUES (1, '/training/alice_sample1.wav', 'Mẫu training 1 của Alice');

-- Tạo 1 lần huấn luyện (training run)
INSERT INTO model_training_runs (model_id, version, total_epochs, status)
VALUES (1, '1.0', 3, 'running');

-- Giả sử log chi tiết cho từng epoch
INSERT INTO training_logs (run_id, epoch, accuracy, loss, log_path)
VALUES 
(1, 1, 0.75, 0.40, '/logs/run1_epoch1.log'),
(1, 2, 0.82, 0.30, '/logs/run1_epoch2.log'),
(1, 3, 0.90, 0.20, '/logs/run1_epoch3.log');
