const bcrypt = require("bcrypt");

async function hashPassword(password) {
    const saltRounds = 10;
    const hashed = await bcrypt.hash(password, saltRounds);
    console.log("Hashed password:", hashed);
    return hashed;
}

// Ví dụ:
hashPassword("123456");
