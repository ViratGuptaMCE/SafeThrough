const express = require('express');
const router = express.Router();
const mapController = require('../controller/maps.controller');
const authMiddleware = require('../middleware/auth');
const { query } = require('express-validator');

router.get('/get-coordinates', query('address').isString().isLength({ min: 3 }), authMiddleware.authUser, mapController.getCoordinates);

router.get('/get-distance-time', query('origin').isString().isLength({ min: 3 }), query('destination').isString().isLength({ min: 3 }), authMiddleware.authUser, mapController.getDistance);

router.get('/get-suggestions', query('input').isString().isLength({ min: 3 }), authMiddleware.authUser, mapController.getSuggestions);

module.exports = router;