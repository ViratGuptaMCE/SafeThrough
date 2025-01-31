const { Server } = require('socket.io');
const userModel = require('./models/user.model');
const captainModel = require('./models/captain.model');

let io;

const initializeSocket = (server) => {
  io = new Server(server, {
    cors: {
      origin: '*',
      methods: ['GET', 'POST']
    }
  });

  io.on('connection', (socket) => {
    console.log(`New client connected: ${socket.id}`);

    socket.on('join', async (data) => {
      const { userId, userType } = data;
      if (userType === 'user') {
        await userModel.findByIdAndUpdate(userId, { socketId: socket.id });

      } else if (userType === 'captain') {
        await captainModel.findByIdAndUpdate(userId, { socketId: socket.id });
      
      }
    });
    socket.on('update-location-captain', async (data) => {
      const { userId, location } = data;
      if (!location || !location.lat || !location.lng) {
        return socket.emit('error',{message : "Invalid location "})
      }
      await captainModel.findByIdAndUpdate(userId, {
        location: {
          lat: location.lat,
          lng : location.lng
      } });
    });
    socket.on('disconnect', () => {
      console.log(`Client disconnected: ${socket.id}`);
    });
  });
};

const sendMessageToSocketId = (socketId, messag) => {
  // console.log(socketId)
  if (io) {
    io.to(socketId).emit(messag.event, messag.data);
    // console.log("msg sent")
  } else {
    console.error('Socket.io is not initialized');
  }
};

module.exports = {
  initializeSocket,
  sendMessageToSocketId
};