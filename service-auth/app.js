const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');

const authRoutes = require('./routes/authRoutes');
const historyRoutes = require('./routes/historyRoutes');

const app = express();
app.use(bodyParser.json());
app.use(cors());

// Routes
app.use('/auth', authRoutes);
app.use('/history', historyRoutes);

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => console.log(`Auth service running on http://localhost:${PORT}`));