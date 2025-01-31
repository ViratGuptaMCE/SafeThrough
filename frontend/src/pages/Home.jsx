import React, { useRef, useState } from "react";
import { useGSAP } from "@gsap/react";
import gsap from "gsap";
import "remixicon/fonts/remixicon.css";
import LocationSearchPanel from "../components/LocationSearchPanel";
import axios from "axios";
import VehiclePanel from "../components/VehiclePanel";
import ConfirmRide from "../components/ConfirmRide";
import LookingForDriver from "../components/LookingForDriver";
import WaitingForDriver from "../components/WaitingForDriver";
// import { set } from "mongoose";
import { SocketContext } from "../context/socketContext";
import { useEffect } from "react";
import { useContext } from "react";
import { UserDataContext } from "../context/UserContext";
// import { Navigate } from "react-router-dom";
import { useNavigate } from "react-router-dom";
import LiveTracking from "../components/LiveTracking";
import { ToLocationContext } from "../context/LocationContext";

const Home = () => {
    const [pickup, setPickup] = useState("");
    const [destination, setDestination] = useState("");
    const [panelOpen, setPanelOpen] = useState(false);
    const vehiclePanelRef = useRef(null);
    const confirmRidePanelRef = useRef(null);
    const vehicleFoundRef = useRef(null);
    const waitingForDriverRef = useRef(null);
    const panelRef = useRef(null);
  const panelCloseRef = useRef(null);
      const [fare, setFare] = useState({});
      const [type, setType] = useState(null);
    const [vehiclePanel, setVehiclePanel] = useState(false);
    const [confirmRidePanel, setConfirmRidePanel] = useState(false);
    const [isPickup, setIsPickup] = useState(true);

    const [vehicleFound, setVehicleFound] = useState(false);
  const [waitingForDriver, setWaitingForDriver] = useState(false);
  const [ride, setRide] = useState(null);
  
  const { socket } = useContext(SocketContext);
  const { user } = useContext(UserDataContext);
  const { toLocation, setToLocation } = useContext(ToLocationContext);
  
  const navigate = useNavigate();
  
  const [toCords, setToCords] = useState(null);

  const geocodeLocation = async (locationName) => {
    const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(
      locationName
    )}`;

    try {
      const response = await fetch(url);
      const data = await response.json();

      if (data.length > 0) {
        const { lat, lon } = data[0]; // Extract the first result
        return [parseFloat(lon), parseFloat(lat)]; // Return as [longitude, latitude]
      } else {
        console.error("No results found for the location:", locationName);
        return null;
      }
    } catch (error) {
      console.error("Error geocoding location:", error);
      return null;
    }
  };
  useEffect(() => {
    if(!user) return;
    socket.emit('join', {userType: 'user', userId: user._id});
  }, [user]);
  
  socket.on('ride-confirmed', ride => {
    setVehicleFound(false)
    setWaitingForDriver(true);
    console.log("We were here");
    setRide(ride);
  })

  socket.on('ride-started', ride => {
    setWaitingForDriver(false);
    console.log("Inside Ride Started")
    navigate('/riding',{state : {ride}});
  })

    const submitHandler = (e) => {
        e.preventDefault()
    }
    const setLocation = (location) => {
      if (isPickup) {
        setPickup(location);
      } else {
        setDestination(location);
      }
  };
  async function createRide() {
    setType("car");
    const response = await axios.post(
      `${import.meta.env.VITE_BASE_URL}/rides/create`,
      {
        pickup,
        destination,
        vehicleType : type,
      },
      {
        headers: {
          Authorization: `Bearer ${localStorage.getItem("token")}`,
        },
      }
    );
    console.log("ride created" , response);
  }
    useGSAP(function () {
        if (panelOpen) {
            gsap.to(panelRef.current, {
                height: '70%',
                padding: 24
                // opacity:1
            })
            gsap.to(panelCloseRef.current, {
                opacity: 1
            })
        } else {
            gsap.to(panelRef.current, {
                height: '0%',
                padding: 0
                // opacity:0
            })
            gsap.to(panelCloseRef.current, {
                opacity: 0
            })
        }
    }, [panelOpen])
    useGSAP(function () {
        if (vehiclePanel) {
            gsap.to(vehiclePanelRef.current, {
                transform: 'translateY(0)'
            })
        } else {
            gsap.to(vehiclePanelRef.current, {
                transform: 'translateY(100%)'
            })
        }
    }, [vehiclePanel])
    useGSAP(function () {
        if (confirmRidePanel) {
            gsap.to(confirmRidePanelRef.current, {
                transform: 'translateY(0)'
            })
        } else {
            gsap.to(confirmRidePanelRef.current, {
                transform: 'translateY(100%)'
            })
        }
    }, [confirmRidePanel])
    useGSAP(function () {
        if (vehicleFound) {
            gsap.to(vehicleFoundRef.current, {
                transform: 'translateY(0)'
            })
        } else {
            gsap.to(vehicleFoundRef.current, {
                transform: 'translateY(100%)'
            })
        }
    }, [vehicleFound])
    useGSAP(function () {
        if (waitingForDriver) {
            gsap.to(waitingForDriverRef.current, {
                transform: 'translateY(0)'
            })
        } else {
            gsap.to(waitingForDriverRef.current, {
                transform: 'translateY(100%)'
            })
        }
    }, [waitingForDriver])
    return (
      <div className="h-screen relative overflow-hidden">
        <img
          className="w-40 absolute left-5 top-5"
          src="/logolucid.png"
          alt=""
        />
        <div className="h-screen w-screen z-100">
          <LiveTracking toLocation={toLocation || [0,0]} />
        </div>
        <div className=" flex flex-col justify-end h-screen absolute top-0 w-full">
          <div className="h-[30%] p-6 bg-white relative">
            <h5
              ref={panelCloseRef}
              onClick={() => {
                setPanelOpen(false);
              }}
              className="absolute opacity-0 right-6 top-6 text-2xl"
            >
              <i className="ri-arrow-down-wide-line"></i>
            </h5>
            <h4 className="text-2xl font-semibold">Get a safe Route</h4>
            <form
              onSubmit={(e) => {
                submitHandler(e);
              }}
            >
              {/* <div className="line absolute h-16 w-1 top-[45%] left-10 bg-gray-700 rounded-full"></div> */}
              <input
                onClick={() => {
                  setPanelOpen(true);
                  setIsPickup(true);
                  // handleLocationClick(true);
                }}
                value={pickup}
                onChange={(e) => {
                  setPickup(e.target.value);
                }}
                className="bg-[#eee] px-12 py-2 text-lg rounded-lg w-full mt-5 hidden"
                type="text"
                placeholder="Start Location"
              />
              <input
                onClick={() => {
                  setPanelOpen(true);
                  // handleLocationClick(true);
                  setIsPickup(false);
                  setPickup("DTU Entrance Gate");
                }}
                value={destination}
                onChange={(e) => {
                  setDestination(e.target.value);
                }}
                className="bg-[#eee] px-12 py-2 text-lg rounded-lg w-full  mt-3"
                type="text"
                placeholder="Enter your destination"
              />
            </form>
            <button
              onClick={async () => {
                setPanelOpen(false);
                setVehiclePanel(true);
                const coord = await geocodeLocation(destination);
                console.log("coord is  ",coord)
                setToLocation(coord);
              }}
              className="bg-black text-white px-4 py-2 rounded-lg mt-4"
            >
              Find Route
            </button>
          </div>
          <div ref={panelRef} className="bg-white h-0">
            <LocationSearchPanel
              setPanelOpen={setPanelOpen}
              setVehiclePanel={setVehiclePanel}
              setLocation={setLocation}
              setPickup={setPickup}
              setToCords={setToCords}
              setDestination={setDestination}
              pickup={pickup}
              destination={destination}
              isPickup={isPickup}
            />
          </div>
        </div>

        <div
          ref={vehiclePanelRef}
          className="fixed w-full z-10 bottom-0 translate-y-full bg-white px-3 py-10 pt-12"
        >
          <VehiclePanel
            setConfirmRidePanel={setConfirmRidePanel}
            setVehiclePanel={setVehiclePanel}
          />
        </div>
        <div
          ref={confirmRidePanelRef}
          className="fixed w-full z-10 bottom-0 translate-y-full bg-white px-3 py-6 pt-12"
        >
          <ConfirmRide
            createRide={createRide}
            setConfirmRidePanel={setConfirmRidePanel}
            setVehicleFound={setVehicleFound}
            pickup={pickup}
            destination={destination}
            fare={{ bike: "₹50", auto: "₹100", car: "₹200" }}
          />
        </div>
        <div
          ref={vehicleFoundRef}
          className="fixed w-full z-10 bottom-0 translate-y-full bg-white px-3 py-6 pt-12"
        >
          <LookingForDriver
            createRide={createRide}
            pickup={pickup}
            destination={destination}
            fare={fare}
            vehicleType={type}
            setVehicleFound={setVehicleFound}
          />
        </div>
        <div
          ref={waitingForDriverRef}
          className="fixed w-full z-10 bottom-0 bg-white px-3 py-6 pt-10"
        >
          <WaitingForDriver
            ride={ride}
            setVehicleFound={setVehicleFound}
            waitingForDriver={waitingForDriver} />
        </div>
      </div>
    );
};

export default Home;
