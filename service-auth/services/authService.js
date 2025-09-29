const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");
const db = require("../utils/db");

const SECRET = "my-secret-key";

exports.login = async (username, password) => {
    const [row] = await db.query("SELECT * FROM users WHERE username = ?", [username]);
    if (row.length == 0) {
        throw new Error("User not found");
    }

    const user = row[0];
    const ismatch = await bcrypt.compare(password, user.password_hash);

    if (!ismatch) {
        throw new Error("Invalid password");
    }

    const token = jwt.sign({ id: user.id, role: user.role }, SECRET, { expiresIn: "1h" })
    return { token, user };
};