const axios = require('axios');
const captainModel = require('../models/captain.model');

module.exports.getAddressCoordinate = async (address) => {
  try {
    const response = await axios.get(`https://maps.gomaps.pro/maps/api/geocode/json?address=${address}&key=${process.env.GOOGLE_MAPS_API_KEY}`);
    if(response.data.status === 'OK') {
      const data = response.data;
      const location = data.results[0].geometry.location;
      return {
        lat: location.lat,
        lng: location.lng
      };
    }else {
      return {message: 'Address not found'};
    }
  }
  catch(err) {
    console.log(err);
  }
}

module.exports.getDistanceTime = async (origin, destination) => {
  if(!origin || !destination) {
    return {message: 'Origin and Destination are required'};
  }
  try {
    const response = await axios.get(`https://maps.gomaps.pro/maps/api/distancematrix/json?origins=${origin}&destinations=${destination}&key=${process.env.GOOGLE_MAPS_API_KEY}`);
    if(response.data.status === 'OK') {
      const data = response.data;
      return {
        distance: data.rows[0].elements[0].distance.text,
        duration: data.rows[0].elements[0].duration.text
      };
    }else {
      return {message: 'Distance not found'};
    }
  }
  catch(err) {
    console.log(err);
    throw new Error(err);
  }
}

module.exports.getSuggestions = async (input) => {
  try {
    const response = await axios.get(`https://maps.gomaps.pro/maps/api/place/autocomplete/json?input=${input}&key=${process.env.GOOGLE_MAPS_API_KEY}`);
    if(response.data.status === 'OK') {
      const data = response.data;
      return data.predictions;
    }else {
      return {message: 'Suggestions not found'};
    }
  }
  catch(err) {
    console.log(err);
  }
}

module.exports.getCaptainsInRadius = async (lat, lng, radius) => {
  const ltd = lat;
  const captains = await captainModel.find({
    location: {
      $geoWithin: {
        $centerSphere: [[ltd, lng], radius / 6371 ]
      }
    }
  });
  return captains;
}