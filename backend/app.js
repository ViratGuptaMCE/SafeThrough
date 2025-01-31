const express = require('express');
const app = express();
const dotenv = require('dotenv');
dotenv.config();
const cors = require('cors');
const userRoutes = require('./routes/user.routes');
const captainRoutes = require('./routes/captain.routes');
const mapRoutes = require('./routes/maps.routes');
const rideRoutes = require('./routes/help.routes');

const cookieParser = require('cookie-parser');

app.use(express.json());
app.use(cookieParser());
app.use(cors());
app.use('/users', userRoutes);
app.use('/captains', captainRoutes);
app.use('/maps', mapRoutes);
app.use('/rides', rideRoutes);
app.use(express.urlencoded({ extended: true }));

const connect = require('./db/db');
connect();

app.get('/', (req, res) => {
  res.send('Hello World!');
});

module.exports  = app;