const express = require('express');
const router = express.Router();
const { body , query} = require('express-validator');
const rideController = require('../controller/help.controller');
const authMiddleware = require('../middleware/auth');

router.post('/create',
  authMiddleware.authUser,
  body('pickup').isString().isLength({ min: 3 }).withMessage('Invalid Origin'),
  body('destination').isString().isLength({ min: 3 }).withMessage('Invalid Destination'),
  body('vehicleType').isString().isIn(['auto', 'bike', 'car']).withMessage('Invalid Vehicle Type'), rideController.createRide
);

router.get('/get-fare',
  authMiddleware.authUser,
  query('pickup').isString().isLength({ min: 3 }).withMessage('Invalid Origin'),
  query('destination').isString().isLength({ min: 3 }).withMessage('Invalid Destination'), rideController.getFare
);

router.post('/confirm', authMiddleware.authCaptain,
  body('rideId').isMongoId().withMessage('Invalid Ride ID'),
  rideController.confirmRide
)


router.get("/start-ride",authMiddleware.authCaptain , query('rideId').isMongoId().withMessage('Invalid ride Id'),rideController.startRide);

router.post(
  "/end-ride",
  authMiddleware.authCaptain,
  body("rideId").isMongoId().withMessage("Invalid ride id"),
  rideController.endRide
);

module.exports = router;