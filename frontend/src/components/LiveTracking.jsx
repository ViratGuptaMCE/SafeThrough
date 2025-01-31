// import React, { useState, useEffect } from "react";

// const containerStyle = {
//   width: "100%",
//   height: "100%",
// };

// const center = {
//   lat: -3.745,
//   lng: -38.523,
// };

// const LiveTracking = () => {
//   const [currentPosition, setCurrentPosition] = useState(center);

//   useEffect(() => {
//     navigator.geolocation.getCurrentPosition((position) => {
//       const { latitude, longitude } = position.coords;
//       setCurrentPosition({
//         lat: latitude,
//         lng: longitude,
//       });
//     });

//     const watchId = navigator.geolocation.watchPosition((position) => {
//       const { latitude, longitude } = position.coords;
//       setCurrentPosition({
//         lat: latitude,
//         lng: longitude,
//       });
//     });

//     return () => navigator.geolocation.clearWatch(watchId);
//   }, []);

//   useEffect(() => {
//     const updatePosition = () => {
//       navigator.geolocation.getCurrentPosition((position) => {
//         const { latitude, longitude } = position.coords;

//         console.log("Position updated:", latitude, longitude);
//         setCurrentPosition({
//           lat: latitude,
//           lng: longitude,
//         });
//       });
//     };

//     updatePosition();

//     const intervalId = setInterval(updatePosition, 1000);

//     return () => clearInterval(intervalId);
//   }, []);

//   return (
//     <div style={containerStyle}>
//       <h2>Live Tracking</h2>
//       <p>
//         Current Position: {currentPosition.lat}, {currentPosition.lng}
//       </p>
//       <div id="map" style={{ width: "100%", height: "100%" }}></div>
//       <script async defer
//         src={`https://maps.gomaps.pro/maps/api/js?key=${import.meta.env.VITE_MAPS_API_KEY}&libraries=geometry,places&callback=initMap`}>
//       </script>
//       <script>
//         let map;
//         function initMap() {
//           map = new gomaps.maps.Map(document.getElementById('map'), {
//             center: {
//               lat: currentPosition.lat, lng: currentPosition.lng,
//             },
//             zoom: 15,
//           });

//           new gomaps.maps.Marker({
//           position: {
//             lat: currentPosition.lat,
//         lng: currentPosition.lng
//           },
//             map: map
//           });
//         }
//       </script>
//     </div>
//   );
// };

// export default LiveTracking;

// import React, { useEffect, useRef, useState } from "react";

// const LiveTracking = () => {
//   const mapRef = useRef(null);
//   const inputRef = useRef(null);
//   const [map, setMap] = useState(null);
//   const [autocomplete, setAutocomplete] = useState(null);
//   const [location, setLocation] = useState(null);

//   useEffect(() => {
//     // Initialize the Google Maps API once the script is loaded
//     const loadScript = (src) => {
//       const script = document.createElement("script");
//       script.src = src;
//       script.async = true;
//       script.defer = true;
//       document.head.appendChild(script);
//       script.onload = () => {
//         initMap();
//       };
//     };

//     loadScript(
//       `https://maps.gomaps.pro/maps/api/js?key=${import.meta.env.VITE_MAPS_API_KEY}&libraries=geometry,places`
//     );

//     // Initialize the map
//     const initMap = () => {
//       const initialMap = new google.maps.Map(mapRef.current, {
//         center: { lat: 28.6520, lng: 77.2315 }, // Default to Sydney
//         zoom: 13,
//       });

//       setMap(initialMap);

//       const input = inputRef.current;
//       const autoComplete = new google.maps.places.Autocomplete(input);
//       autoComplete.bindTo("bounds", initialMap);

//       setAutocomplete(autoComplete);

//       // Add listener for place changes
//       autoComplete.addListener("place_changed", () => {
//         const place = autoComplete.getPlace();
//         if (!place.geometry) {
//           console.log(
//             "No details available for the input: '" + place.name + "'"
//           );
//           return;
//         }

//         setLocation({
//           lat: place.geometry.location.lat(),
//           lng: place.geometry.location.lng(),
//         });

//         if (place.geometry.viewport) {
//           initialMap.fitBounds(place.geometry.viewport);
//         } else {
//           initialMap.setCenter(place.geometry.location);
//           initialMap.setZoom(17);
//         }

//         // Place a marker on the selected location
//         new google.maps.Marker({
//           position: place.geometry.location,
//           map: initialMap,
//         });
//       });
//     };

//     // Clean up the map when the component unmounts
//     return () => {
//       if (map) {
//         google.maps.event.clearInstanceListeners(map);
//       }
//     };
//   }, []);

//   useEffect(() => {
//     if (map && location) {
//       // If location state changes, update map center
//       map.setCenter(location);
//       new google.maps.Marker({
//         position: location,
//         map: map,
//       });
//     }
//   }, [map, location]);

//   return (
//     <div>
//       <input
//         ref={inputRef}
//         id="pac-input"
//         type="text"
//         placeholder="Search for a place"
//         style={{
//           marginTop: "10px",
//           width: "300px",
//           padding: "5px",
//           fontSize: "14px",
//         }}
//       />
//       <div
//         id="map"
//         ref={mapRef}
//         style={{
//           height: "400px",
//           width: "100%",
//         }}
//       />
//     </div>
//   );
// };

// export default LiveTracking;


// import React, { useEffect, useRef, useState } from "react";

// const LiveTracking = () => {
//   const mapRef = useRef(null);
//   const inputRef = useRef(null);
//   const [map, setMap] = useState(null);
//   const [autocomplete, setAutocomplete] = useState(null);
//   const [location, setLocation] = useState(null);
//   const [marker, setMarker] = useState(null); // State to store the marker

//   useEffect(() => {
//     // Initialize the Google Maps API once the script is loaded
//     const loadScript = (src) => {
//       const script = document.createElement("script");
//       script.src = src;
//       script.async = true;
//       script.defer = true;
//       document.head.appendChild(script);
//       script.onload = () => {
//         initMap();
//       };
//     };

//     loadScript(
//       `https://maps.gomaps.pro/maps/api/js?key=${
//         import.meta.env.VITE_MAPS_API_KEY
//       }&libraries=geometry,places`
//     );

//     // Initialize the map
//     const initMap = () => {
//       const initialMap = new google.maps.Map(mapRef.current, {
//         center: { lat: 28.652, lng: 77.2315 }, // Default to Sydney
//         zoom: 13,
//       });

//       setMap(initialMap);

//       const input = inputRef.current;
//       const autoComplete = new google.maps.places.Autocomplete(input);
//       autoComplete.bindTo("bounds", initialMap);

//       setAutocomplete(autoComplete);

//       // Add listener for place changes
//       autoComplete.addListener("place_changed", () => {
//         const place = autoComplete.getPlace();
//         if (!place.geometry) {
//           console.log(
//             "No details available for the input: '" + place.name + "'"
//           );
//           return;
//         }

//         setLocation({
//           lat: place.geometry.location.lat(),
//           lng: place.geometry.location.lng(),
//         });

//         if (place.geometry.viewport) {
//           initialMap.fitBounds(place.geometry.viewport);
//         } else {
//           initialMap.setCenter(place.geometry.location);
//           initialMap.setZoom(17);
//         }

//         // Place a marker on the selected location
//         new google.maps.Marker({
//           position: place.geometry.location,
//           map: initialMap,
//         });
//       });
//     };

//     // Clean up the map when the component unmounts
//     return () => {
//       if (map) {
//         google.maps.event.clearInstanceListeners(map);
//       }
//     };
//   }, []);

//   useEffect(() => {
//     if (map && location) {
//       // If location state changes, update map center
//       map.setCenter(location);
//       new google.maps.Marker({
//         position: location,
//         map: map,
//       });
//     }
//   }, [map, location]);

//   useEffect(() => {
//     if (navigator.geolocation) {
//       const geoSuccess = (position) => {
//         const { latitude, longitude } = position.coords;
//         const currentLocation = { lat: latitude, lng: longitude };

//         setLocation(currentLocation);
//         map.setCenter(currentLocation);

//         const currentMarker = new google.maps.Marker({
//           position: currentLocation,
//           map: map,
//         });

//         setMarker(currentMarker);

//         // Update the marker every 1 second
//         const interval = setInterval(() => {
//           if (navigator.geolocation) {
//             navigator.geolocation.getCurrentPosition((position) => {
//               const { latitude, longitude } = position.coords;
//               const updatedLocation = { lat: latitude, lng: longitude };
              
//               // Update marker position
//               currentMarker.setPosition(updatedLocation);
//               map.setCenter(updatedLocation);
//             });
//           }
//         }, 1000);

//         // Clear interval when component unmounts
//         return () => clearInterval(interval);
//       };

//       const geoError = () => {
//         console.log("Geolocation not available or permission denied.");
//       };

//       navigator.geolocation.getCurrentPosition(geoSuccess, geoError);
//     }
//   }, [map]);

//   return (
//     <div>
//       <input
//         ref={inputRef}
//         id="pac-input"
//         type="text"
//         placeholder="Search for a place"
//         style={{
//           marginTop: "10px",
//           width: "300px",
//           padding: "5px",
//           fontSize: "14px",
//         }}
//       />
//       <div
//         id="map"
//         ref={mapRef}
//         style={{
//           height: "400px",
//           width: "100%",
//         }}
//       />
//     </div>
//   );
// };

// export default LiveTracking;


import React, { useEffect, useRef, useState } from "react";
import "ol/ol.css"; // OpenLayers CSS
import { Map, View } from "ol";
import TileLayer from "ol/layer/Tile";
import OSM from "ol/source/OSM"; // OpenStreetMap
import { fromLonLat } from "ol/proj";
import { Point, LineString } from "ol/geom";
import { Feature } from "ol";
import { Vector as VectorLayer } from "ol/layer";
import { Vector as VectorSource } from "ol/source";
import { Style, Icon, Stroke } from "ol/style";

/**
 * Function to display a route between two locations on the map.
 * @param {Array<number>} from - The "from" location as [longitude, latitude].
 * @param {Array<number>} to - The "to" location as [longitude, latitude].
 * @param {VectorSource} routeVectorSource - The OpenLayers vector source for the route layer.
 * @param {Map} map - The OpenLayers map instance.
 */
const showRoute = async (from, to, routeVectorSource, map) => {
  const fromCoords = from;
  const toCoords = to;

  // OSRM API URL for route calculation
  const url = `https://router.project-osrm.org/route/v1/driving/${fromCoords[0]},${fromCoords[1]};${toCoords[0]},${toCoords[1]}?overview=full&geometries=geojson`;

  try {
    const response = await fetch(url);
    const data = await response.json();

    if (data.routes && data.routes.length > 0) {
      const route = data.routes[0];
      const routeCoordinates = route.geometry.coordinates;

      // Convert coordinates to OpenLayers format
      const formattedCoordinates = routeCoordinates.map((coord) =>
        fromLonLat(coord)
      );

      // Create a LineString feature for the route
      const routeFeature = new Feature({
        geometry: new LineString(formattedCoordinates),
      });

      // Style the route
      routeFeature.setStyle(
        new Style({
          stroke: new Stroke({
            color: "blue",
            width: 3,
          }),
        })
      );

      // Clear previous route and add the new one
      routeVectorSource.clear();
      routeVectorSource.addFeature(routeFeature);

      // Fit the map view to the route
      map.getView().fit(routeFeature.getGeometry(), {
        padding: [50, 50, 50, 50],
        maxZoom: 15,
      });
    } else {
      console.error("No route found.");
    }
  } catch (error) {
    console.error("Error calculating route:", error);
  }
};

const LiveTracking = (props) => {
  const mapRef = useRef(null);
  const [map, setMap] = useState(null);
  const [routeVectorSource, setRouteVectorSource] = useState(
    new VectorSource()
  );
  const [markerVectorSource, setMarkerVectorSource] = useState(
    new VectorSource()
  );
  const [fromLocation, setFromLocation] = useState(null);

  // Initialize the map
  useEffect(() => {
    const initialMap = new Map({
      target: mapRef.current,
      layers: [
        new TileLayer({
          source: new OSM(), // Use OpenStreetMap as the base layer
        }),
        new VectorLayer({
          source: routeVectorSource, // Layer for the route
        }),
        new VectorLayer({
          source: markerVectorSource, // Layer for markers
        }),
      ],
      view: new View({
        center: fromLonLat([77.2315, 28.652]), // Default to Delhi
        zoom: 13,
      }),
    });

    setMap(initialMap);

    // Cleanup on unmount
    return () => {
      initialMap.setTarget(null);
    };
  }, []);

  // Fetch the user's current location and handle route calculation
  useEffect(() => {
    if (!map) return;

    navigator.geolocation.getCurrentPosition(
      (position) => {
        const { latitude, longitude } = position.coords;
        const currentLocation = fromLonLat([longitude, latitude]);
        setFromLocation(currentLocation);

        // Add a marker for the current location
        const fromMarker = new Feature({
          geometry: new Point(currentLocation),
        });
        fromMarker.setStyle(
          new Style({
            image: new Icon({
              src: "https://openlayers.org/en/latest/examples/data/icon.png", // Marker icon
              scale: 0.5,
            }),
          })
        );
        markerVectorSource.addFeature(fromMarker);

        // Check if props.toLocation is provided
        if (
          props.toLocation &&
          props.toLocation[0] !== 0 &&
          props.toLocation[1] !== 0
        ) {
          // Calculate and display the route
          showRoute(
            [longitude, latitude],
            props.toLocation,
            routeVectorSource,
            map
          );
        } else {
          // Center the map on the user's current location
          map.getView().setCenter(currentLocation);
          map.getView().setZoom(15);
        }
      },
      (error) => {
        console.error("Error getting current location:", error);
      }
    );
  }, [map, props.toLocation]);

  return (
    <div>
      <div
        ref={mapRef}
        style={{
          height: "600px",
          width: "100%",
        }}
      />
    </div>
  );
};

export default LiveTracking;