const { model } = require('mongoose');
const userModel = require('../models/user.model');
const { validationResult } = require('express-validator');
 
const userService = require('../services/user.service');

const blackListToken = require('../models/blackListToken.model');


module.exports.registerUser = async (req, res, next) => {
  const err = validationResult(req);
  if (!err.isEmpty()) {
    return res.status(400).send({ message: err.array() });
  }
  // console.log(req.body);
  const { email, password, fullName } = req.body;

  const hashedPassword = await userModel.hashPassword(password);

  const user = await userService.createUser({ email, password: hashedPassword, firstName : fullName.firstName, lastName : fullName.lastName });

  const token = user.generateAuthToken();

  res.status(201).send({ user, token });
}


module.exports.loginUser = async (req, res, next) => {
  const err = validationResult(req);
  if (!err.isEmpty()) {
    return res.status(400).send({ message: err.array() });
  }
  const { email, password} = req.body;
  // console.log(email, password);
  const user = await userModel.findOne({ email }).select('+password');
  console.log(user);
  if (!user) {
    return res.status(401).send({ message: ' not valid email or password' });
  }
  
  const isMatch = await user.comparePassword(password);

  if (!isMatch) {
    return res.status(401).send({ message: 'Invalid email or pass' });
  }
  
  const token = user.generateAuthToken();

  res.cookie('token', token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    maxAge: 7 * 24 * 60 * 60 * 1000,
  });

  res.status(201).send({ user, token });
}


module.exports.getProfile = async (req, res, next) => {
  res.status(200).json(req.user);
}

module.exports.logoutUser = async (req, res, next) => {
  res.clearCookie('token');
  const token = req.cookies.token || req.headers.authorization?.split(' ')[1];
  console.log(token);
  await blackListToken.create({ token });
  res.status(200).json({ message: 'Logged out' });
}