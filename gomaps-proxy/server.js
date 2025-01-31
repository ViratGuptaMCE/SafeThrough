const express = require("express");
const axios = require("axios");
const app = express();
const port = 3000;

app.get("/maps", async (req, res) => {
  const { lat, lng, zoom, size, key } = req.query;
  try {
    const response = await axios.get(
      `https://maps.gomaps.pro/maps/api/staticmap`,
      {
        params: {
          center: `${lat},${lng}`,
          zoom,
          size,
          key,
        },
      }
    );
    res.send(response.data);
  } catch (error) {
    res.status(500).send("Error fetching map data");
  }
});

app.listen(port, () => {
  console.log(`Proxy server running at http://localhost:${port}`);
});
