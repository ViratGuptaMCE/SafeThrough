const mapService = require('../services/maps.service');
const { validationResult } = require('express-validator');


module.exports.getCoordinates = async (req, res) => {
  const err = validationResult(req);
  if (!err.isEmpty()) {
    return res.status(400).json({ message: err.array() });
  }
  const { address } = req.query;
  try {
    const coordinates = await mapService.getAddressCoordinate(address);
    res.status(200).json(coordinates);
  }catch(err) {
    res.status(404).json({ message: 'Coordinates not found' });
  }
}

module.exports.getDistance = async (req, res) => {
  const err = validationResult(req);
  if (!err.isEmpty()) {
    return res.status(400).json({ message: err.array() });
  }
  const { origin, destination } = req.query;
  try {
    const distanceTime = await mapService.getDistanceTime(origin, destination);
    res.status(200).json(distanceTime);
  } catch (err) {
    res.status(404).json({ message: 'Distance not found' });
  }
}

module.exports.getSuggestions = async (req, res) => {
  const err = validationResult(req);
  if (!err.isEmpty()) {
    return res.status(400).json({ message: err.array() });
  }
  const { input } = req.query;
  try {
    const suggestions = await mapService.getSuggestions(input);
    res.status(200).json(suggestions);
  } catch (err) {
    res.status(404).json({ message: 'Suggestions not found' });
  }
}