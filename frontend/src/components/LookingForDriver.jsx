import React from "react";
const LookingForDriver = (props) => {
  return (
    <div>
      <h5
        className="p-1 text-center w-[93%] absolute top-0"
        onClick={() => {
          props.setVehicleFound(false);
        }}
      >
        <i className="text-3xl text-gray-200 ri-arrow-down-wide-line"></i>
      </h5>
      <h3 className="text-2xl font-semibold mb-5">Help On Way</h3>
      <div className="flex gap-2 justify-between flex-col items-center">
        <img
          className="h-20"
          src="/namaste.png"
          alt=""
        />
        <div className="w-full mt-5">
          <div className="flex items-center gap-5 p-3 border-b-2">
            <i className="ri-map-pin-user-fill"></i>
            <div>
              <h3 className="text-lg font-medium">Requests Send to nearby</h3>
              <p className="text-sm mt-4 text-gray-600">
                {/* Came from 400m away */}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-5 p-3 border-b-2">
            <i className="text-lg ri-map-pin-2-fill"></i>
            <div>
              <h3 className="text-lg font-medium">Police is being Informed</h3>
              <p className="text-sm mt-4 text-gray-600">
                {/* Came from 750m away */}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-5 p-3">
            <i className="ri-currency-line"></i>
            <div>
              <h3 className="text-lg font-medium">Family members got notified</h3>
              {/* <p className="text-sm -mt-1 text-gray-600">Took 12 mins to arrive</p> */}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
export default LookingForDriver;
