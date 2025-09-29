const historyService = require("../services/historyService");

exports.getHistory = async (req, res) => {
    try {
        const userId = req.params.userId;
        const history = await historyService.history(userId);
        res.json({ success: true, history });
    } catch (error) {
        res.status(500).json({ success: false, message: error.message });
    }
}