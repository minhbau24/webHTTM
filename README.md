# WEBHTTM - Voice Authentication System

🚧 **Dự án đang trong quá trình phát triển** 🚧

Hệ thống xác thực bằng giọng nói sử dụng công nghệ AI và machine learning để nhận dạng người dùng thông qua đặc trưng giọng nói.

## 📋 Mục lục
- [Tổng quan](#tổng-quan)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Công nghệ sử dụng](#công-nghệ-sử-dụng)
- [Cài đặt và chạy dự án](#cài-đặt-và-chạy-dự-án)
- [API Documentation](#api-documentation)
- [Tính năng](#tính-năng)
- [Roadmap](#roadmap)

## 🎯 Tổng quan

WEBHTTM là hệ thống xác thực đa phương thức kết hợp:
- **Xác thực truyền thống**: Username/Password
- **Xác thực giọng nói**: Sử dụng AI để nhận dạng đặc trưng giọng nói
- **Anti-spoofing**: Chống giả mạo giọng nói
- **Quản lý lịch sử**: Theo dõi các lần đăng nhập

## 📁 Cấu trúc dự án

```
WEBHTTM/
├── frontend/                    # Giao diện người dùng (HTML/CSS/JS)
│   ├── index.html              # Trang chủ
│   ├── login.html              # Trang đăng nhập
│   ├── register.html           # Trang đăng ký
│   ├── history.html            # Trang lịch sử đăng nhập
│   ├── admin.html              # Trang quản trị
│   └── js/
│       └── main.js             # JavaScript chính
│
├── service-auth/               # Service xác thực (Node.js + Express)
│   ├── controllers/            # Xử lý logic nghiệp vụ
│   │   ├── authController.js   # Controller xác thực
│   │   └── historyController.js # Controller lịch sử
│   ├── services/               # Business logic
│   │   ├── authService.js      # Service xác thực
│   │   └── historyService.js   # Service lịch sử
│   ├── models/                 # Data models
│   │   ├── User.js             # Model người dùng
│   │   └── History.js          # Model lịch sử
│   ├── routes/                 # API routes
│   │   ├── authRoutes.js       # Routes xác thực
│   │   └── historyRoutes.js    # Routes lịch sử
│   ├── utils/                  # Tiện ích
│   │   └── db.js               # Kết nối database
│   ├── app.js                  # Entry point Express
│   └── package.json            # Dependencies Node.js
│
└── service-voice/              # Service xử lý giọng nói (Python + FastAPI)
    ├── api/                    # API endpoints
    │   ├── voice_register.py   # API đăng ký giọng nói
    │   ├── voice_verify.py     # API xác thực giọng nói
    │   └── model_manage.py     # API quản lý model
    ├── core/                   # Core AI/ML logic
    │   ├── inference.py        # Suy luận model
    │   ├── training.py         # Huấn luyện model
    │   └── model_loader.py     # Tải model
    ├── services/               # AI Services
    │   ├── anti_spoof.py       # Chống giả mạo
    │   └── speaker_id.py       # Nhận dạng người nói
    ├── utils/                  # Tiện ích AI
    │   ├── audio_preprocess.py # Tiền xử lý âm thanh
    │   └── feature_extraction.py # Trích xuất đặc trưng
    ├── main.py                 # Entry point FastAPI
    └── requirements.txt        # Dependencies Python
```

### sơ đồ tổng quan của hệ thống
```
                    🌐 WEBHTTM VOICE AUTHENTICATION SYSTEM 🌐
                              
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           📱 PRESENTATION LAYER (Frontend)                      │
│                                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ login.html  │  │register.html│  │ admin.html  │  │history.html │             │
│  │   🔐 Login │  │  📝 Đăng ký │  │ 👨‍💼 Quản trị │  │ 📊 Lịch sử │             │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘             │
│                          │                  │                                   │
│                          └──────┬───────────┘                                   │
│                                 │ 🎤 Voice + 📝 Form Data                      │
└─────────────────────────────────┼───────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │      🌍 HTTP/HTTPS        │
                    │      (JSON APIs)          │
                    └─────────────┬─────────────┘
                                  │
┌─────────────────────────────────┼─────────────────────────────────────────────┐
│                    🔧 APPLICATION LAYER (Backend Services)                    │
│                                 │                                             │
│  ┌──────────────────────────────┼──────────────────────────────────────────┐  │
│  │        🔑 AUTH SERVICE       │           🎵 VOICE SERVICE              │  │
│  │       (Node.js + Express)    │          (Python + FastAPI)              │  │
│  │          Port: 3000          │             Port: 8000                   │  │
│  │                              │                                          │  │
│  │  ┌─────────────────────┐     │     ┌─────────────────────────────┐      │  │
│  │  │   🛣️ Routes         │     │     │      🤖 AI/ML Core         │      │  │
│  │  │ • /auth/login       │     │     │ • Speaker Recognition       │      │  │
│  │  │ • /auth/register    │     │     │ • Anti-spoofing             │      │  │
│  │  │ • /history/:userId  │     │     │ • Voice Embeddings          │      │  │
│  │  └─────────────────────┘     │     └─────────────────────────────┘      │  │
│  │                              │                                          │  │
│  │  ┌─────────────────────┐     │     ┌─────────────────────────────┐      │  │
│  │  │   🧠 Controllers    │     │     │      📡 API Endpoints      │      │  │
│  │  │ • authController    │◄────┼────►│ • /voice/register          │       │  │
│  │  │ • historyController │     │     │ • /voice/verify            │       │  │
│  │  └─────────────────────┘     │     │ • /model/train             │       │  │
│  │                              │     │ • /model/status            │       │  │
│  │  ┌─────────────────────┐     │     └─────────────────────────────┘      │  │
│  │  │   ⚙️ Services       │     │                                          │  │
│  │  │ • authService       │     │     ┌─────────────────────────────┐      │  │
│  │  │ • historyService    │     │     │      🔊 Audio Processing    │      │  │
│  │  │ • JWT tokens        │     │     │ • Preprocessing             │      │  │
│  │  │ • Password hashing  │     │     │ • Feature Extraction        │      │  │
│  │  └─────────────────────┘     │     │ • MFCC, Mel-spectrogram     │      │  │
│  └──────────────────────────────┘     └─────────────────────────────┘      │  │
│                                                                            │  │
└───────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  │ 🔄 Database Queries (SQL)
                                  │
┌─────────────────────────────────┼─────────────────────────────────────────────┐
│                      💾 DATA LAYER (Database)                                 │
│                         MySQL 8.0 (Port: 3306)                                │
│                                 │                                             │
│   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐             │
│   │  users      │ │voice_samples│ │  auth_logs  │ │   models    │             │
│   │ • id        │ │ • user_id   │ │ • user_id   │ │ • model_id  │             │
│   │ • username  │ │ • audio_path│ │ • result    │ │ • model_path│             │
│   │ • email     │ │ • embedding │ │ • timestamp │ │ • accuracy  │             │
│   │ • password  │ │ • created_at│ │ • is_success│ │ • created_at│             │
│   └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘             │
│                                                                               │
│   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐             │
│   │training_data│ │training_runs│ │training_logs│ │system_config│             │
│   │ • dataset_id│ │ • run_id    │ │ • epoch     │ │ • config_key│             │
│   │ • user_id   │ │ • model_id  │ │ • loss      │ │ • config_val│             │
│   │ • file_path │ │ • status    │ │ • accuracy  │ │ • updated_at│             │
│   │ • label     │ │ • started_at│ │ • timestamp │ │             │             │
│   └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘             │
└───────────────────────────────────────────────────────────────────────────────┘
```
## 🛠 Công nghệ sử dụng

### Frontend
- **HTML5/CSS3/JavaScript**: Giao diện người dùng
- **Fetch API**: Gọi API

### Backend - Service Auth (Node.js)
- **Express.js**: Web framework
- **MySQL 8.0+**: Database với UTF8MB4 encoding
- **bcrypt**: Mã hóa password
- **JWT**: Token authentication
- **CORS**: Cross-origin requests

### Backend - Service Voice (Python)
- **FastAPI**: Web framework hiệu năng cao
- **PyTorch**: Deep learning framework cho speaker identification
- **librosa**: Xử lý và phân tích âm thanh
- **scikit-learn**: Machine learning algorithms
- **NumPy**: Tính toán khoa học và xử lý vector embeddings

### Database Design
- **8 bảng chính**: Users, Voice Samples, Auth Logs, Models, Training Data
- **BLOB storage**: Lưu trữ embedding vectors
- **Foreign Key constraints**: Đảm bảo data integrity
- **Sample data**: Có sẵn user test và model data

## 🚀 Cài đặt và chạy dự án

### Yêu cầu hệ thống
- **Node.js** >= 16.0.0
- **Python** >= 3.8
- **MySQL** >= 8.0
- **npm** hoặc **yarn**

### 1. Clone dự án
```bash
git clone <repository-url>
cd WEBHTTM
```

### 2. Thiết lập Database
Dự án sử dụng file `schema.sql` để tạo cấu trúc database hoàn chỉnh:

```bash
# Import schema vào MySQL
mysql -u root -p < schema.sql
```

**Cấu trúc Database bao gồm:**
- **users**: Tài khoản người dùng với thông tin cơ bản
- **voice_samples**: Lưu trữ mẫu giọng nói gốc và embedding vectors
- **auth_logs**: Lịch sử xác thực chi tiết (thành công/thất bại)
- **models**: Quản lý các AI models (SpeakerID, Anti-spoofing)
- **training_datasets**: Dữ liệu training cho từng user
- **model_training_runs**: Theo dõi quá trình huấn luyện model
- **training_logs**: Log chi tiết theo epoch/accuracy/loss

Database được thiết kế với UTF8MB4 encoding và sample data sẵn có.

### 3. Chạy Service Auth (Node.js)
```bash
# Di chuyển vào thư mục service-auth
cd service-auth

# Cài đặt dependencies
npm install

# Cấu hình database trong utils/db.js
# Sửa thông tin: host, user, password, database

# Chạy service (development)
npm run dev

# Hoặc chạy production
npm start
```

Service sẽ chạy tại: `http://localhost:3000`

### 4. Chạy Service Voice (Python) - Đang phát triển
```bash
# Di chuyển vào thư mục service-voice
cd service-voice

# Tạo virtual environment
python -m venv venv

# Kích hoạt virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt

# Chạy service
python main.py
```

Service sẽ chạy tại: `http://localhost:8000`

### 5. Chạy Frontend
Mở file `frontend/login.html` trong trình duyệt hoặc sử dụng live server.

**Thông tin đăng nhập test (có sẵn trong schema.sql):**
- Username: `alice`
- Email: `alice@example.com`
- Password: `123456` (hash: `$2b$10$YwysaxJ33/5t7cYdk9iA/uVLgE6aiYYlbZuk2n.R4ObgSc7miqKeu`)

**Note**: Bạn cần cập nhật auth service để sử dụng database thực tế thay vì mock data.

## 📖 API Documentation

### Auth Service (Node.js) - Port 3000

#### POST /auth/login
Đăng nhập người dùng
```json
{
  "username": "alice",
  "password": "123456"
}
```

#### GET /auth/logs/:userId
Lấy lịch sử xác thực của user (từ bảng auth_logs)

### Voice Service (Python) - Port 8000 (Đang phát triển)

#### POST /voice/register
Đăng ký giọng nói - lưu vào bảng `voice_samples`
```json
{
  "user_id": 1,
  "audio_file": "base64_encoded_audio"
}
```

#### POST /voice/verify  
Xác thực giọng nói - ghi log vào bảng `auth_logs`
```json
{
  "user_id": 1,
  "audio_file": "base64_encoded_audio"
}
```

#### GET /model/status
Trạng thái các AI models từ bảng `models`

#### POST /model/train
Khởi tạo quá trình training model mới - tạo record trong `model_training_runs`

#### GET /training/logs/:run_id
Lấy chi tiết training logs theo run_id

## ✨ Tính năng

### ✅ Đã hoàn thành
- [x] Database schema hoàn chỉnh (8 bảng với sample data)
- [x] Xác thực username/password cơ bản
- [x] JWT token authentication  
- [x] CORS support
- [x] Database connection (MySQL)
- [x] Cấu trúc API routes cho auth và history
- [x] Giao diện đăng nhập cơ bản

### 🚧 Đang phát triển
- [ ] Tích hợp database thực tế vào auth service (hiện đang dùng mock data)
- [ ] API xử lý voice samples và embeddings
- [ ] Anti-spoofing detection algorithms
- [ ] Model training pipeline với logging
- [ ] Giao diện đăng ký giọng nói
- [ ] Dashboard admin để xem training logs và model status
- [ ] Real-time voice processing
- [ ] Audio preprocessing và feature extraction

### 📅 Roadmap
- [ ] **Phase 1**: Tích hợp database với auth service
- [ ] **Phase 2**: Implement voice registration và verification APIs
- [ ] **Phase 3**: Model training pipeline với real-time monitoring
- [ ] **Phase 4**: Advanced anti-spoofing algorithms
- [ ] **Phase 5**: Admin dashboard với training metrics
- [ ] **Phase 6**: Mobile app support
- [ ] **Phase 7**: Docker containerization
- [ ] **Phase 8**: CI/CD pipeline và performance optimization

## 🤝 Đóng góp

Dự án đang trong giai đoạn phát triển. Mọi đóng góp đều được chào đón!

1. Fork dự án
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## 📝 License

Dự án này được phân phối dưới giấy phép MIT. Xem file `LICENSE` để biết thêm chi tiết.

## 📞 Liên hệ

- **Email**: [your-email@domain.com]
- **GitHub**: [your-github-username]

---

⭐ **Nếu dự án hữu ích, hãy cho một star nhé!** ⭐