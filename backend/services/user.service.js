const userModel = require('../models/user.model');


module.exports.createUser = async (
  { email, password, firstName , lastName })=> {
  if (!email || !password || !firstName) {
    return res.status(400).send({ message: 'Please provide all required fields' });
  }
  const user = new userModel({ email, password, fullName: { firstName, lastName } });
  user.save();
  return user;
}