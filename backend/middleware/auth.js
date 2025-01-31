const userModel = require('../models/user.model');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken'); 
const blackListToken = require('../models/blackListToken.model'); 
const captainModel = require('../models/captain.model');
const blackListTokenModel = require('../models/blackListToken.model');

module.exports.authUser = async (req, res, next) => {
  const token = req.header('Authorization')?.split(' ')[1] || req.cookies.token;
  if(!token) {
    return res.status(401).json({message: 'Unauthorized'});
  }

  const isBlackListed = await blackListTokenModel.findOne({ token : token });
  if (isBlackListed) {
    return res.status(401).json({message: 'Unauthorized'});
  }
  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    const user = await userModel.findById(decoded._id);
    req.user = user;
    return next();
  } catch (err) {
    return res.status(401).json({message: 'Unauthorized'});
  }
}


module.exports.authCaptain = async (req, res, next) => {
  const token = req.header('Authorization')?.split(' ')[1] || req.cookies.token;
  if(!token) {
    return res.status(401).json({message: 'Unauthorized'});
  }

  const isBlackListed = await blackListTokenModel.findOne({ token : token });
  if (isBlackListed) {
    return res.status(401).json({message: 'Unauthorized'});
  }
  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    const captain = await captainModel.findById(decoded._id);
    req.captain = captain;
    return next();
  } catch (err) {
    return res.status(401).json({message: 'Unauthorized access'});
  } 
}