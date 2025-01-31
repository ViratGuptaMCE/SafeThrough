import React from "react";
const ConfirmRide = (props) => {
  return (
    <div>
      <h5
        className="p-1 text-center w-[93%] absolute top-0"
        onClick={() => {
          props.setConfirmRidePanel(false);
        }}
      >
        <i className="text-3xl text-gray-200 ri-arrow-down-wide-line"></i>
      </h5>
      <h3 className="text-2xl font-semibold mb-5">Ask for Help</h3>
      <div className="flex gap-2 justify-between flex-col items-center">
        <img
          className="h-20"
          src="/peace.png"
          alt=""
        />
        <div className="w-full mt-5">
          <div className="flex items-center gap-5 p-3 border-b-2">
            <i className="ri-map-pin-user-fill"></i>
            <div>
              <h3 className="text-lg font-medium">3 mins</h3>
              <p className="text-sm -mt-1 text-gray-600">
                {/* {props.pickup} */}
                Usual time for someone respond to help
              </p>
            </div>
          </div>
          <div className="flex items-center gap-5 p-3 border-b-2">
            <i className="text-lg ri-map-pin-2-fill"></i>
            <div>
              <h3 className="text-lg font-medium">Location Privacy</h3>
              <p className="text-[10px] mt-1 text-green-600">
                {/* {props.destination} */}
                Don't worry your location will only be visible to
                someone when they accept request and information will be sent to police also.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-5 p-3">
            <i className="ri-currency-line"></i>
            <div>
              <h3 className="text-lg font-medium">{props.fare[props.vehicleType ]} </h3>
              <p className="text-sm -mt-1 text-gray-600">Usual Police Response : 7 mins away</p>
            </div>
          </div>
        </div>
        <button onClick={() => {
          props.setVehicleFound(true)
          props.setConfirmRidePanel(false)
          props.createRide()
        }} className="w-full mt-5 bg-green-600 text-white font-semibold p-2 rounded-lg">
          Raise Request
        </button>
      </div>
    </div>
  );
};
export default ConfirmRide;