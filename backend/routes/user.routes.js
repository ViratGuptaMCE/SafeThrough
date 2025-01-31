const express = require('express');
const router = express.Router();
const userController = require('../controller/user.controller');
const { body } = require('express-validator');
const authMiddleware = require('../middleware/auth');

router.post('/register', [
    body('email').isEmail(),
    body('password').isLength({min: 6}).withMessage('Password should be at least 6 characters'),
], userController.registerUser);

router.post(
  "/login",
  [
    body("email").isEmail(),
    body("password")
      .isLength({ min: 6 })
      .withMessage("Password should be at least 6 characters"),
  ],
  userController.loginUser
);

router.get('/profile', authMiddleware.authUser, userController.getProfile);
router.get('/logout',authMiddleware.authUser, userController.logoutUser);

module.exports = router;