const mysql = require("mysql2/promise");

const pool = mysql.createPool({
    host: "localhost",
    user: "root",
    password: "123456",
    database: "voice_auth"
})

module.exports = pool;