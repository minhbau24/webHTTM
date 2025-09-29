const db = require("../utils/db");

exports.history = async (userId) => {
    const [rows] = await db.query(
        "SELECT id, audio_input_path, is_real_voice, matched_user_id, result, timestamp \
         FROM auth_logs WHERE user_id = ? ORDER BY timestamp DESC",
        [userId]
    );

    if (rows.length === 0) {
        throw new Error("No history found for this user");
    }

    return rows;
};
