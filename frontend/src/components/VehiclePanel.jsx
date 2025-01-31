import React from "react";
const VehiclePanel = (props) => {
  return (
    <div>
      <h5
        className="p-1 text-center w-[93%] absolute top-0"
        onClick={() => {
          props.setVehiclePanel(false);
        }}
      >
        <i className="text-3xl text-gray-200 ri-arrow-down-wide-line"></i>
      </h5>
      <h3 className="text-2xl font-semibold mb-5">Any Emergency</h3>
      <div
        onClick={() => {
          props.setConfirmRidePanel(true);
        }}
        className="flex border-2 active:border-black  mb-2 rounded-xl w-full p-3  items-center justify-between"
      >
        <img className="h-10" src="/suspicious.png" alt="" />
        <div className="ml-2 w-1/2">
          <h4 className="font-medium text-base text-orange-500">
            Suspicious <span>{/* <i className="ri-user-3-fill"></i> */}</span>
          </h4>
          <h5 className="font-medium text-sm">Track Voice </h5>
          <p className="font-normal text-xs text-gray-600">
            We will just track sound signals and act accordingly in case of any
            emergency
          </p>
        </div>
        <button className="text-lg bg-amber-400 p-2 px-4 rounded-3xl font-semibold">Confirm</button>
      </div>
      <div
        onClick={() => {
          props.setConfirmRidePanel(true);
        }}
        className="flex border-2 active:border-black mb-2 rounded-xl w-full p-3  items-center justify-between"
      >
        <img className="h-10" src="/sos.png" alt="" />
        <div className="-ml-2 w-1/2">
          <h4 className="font-medium text-base text-red-500">
            SOS <span>{/* <i className="ri-user-3-fill"></i>1 */}</span>
          </h4>
          <h5 className="font-medium text-sm">Send Location </h5>
          <p className="font-normal text-xs text-gray-600">
            To Police and family
          </p>
        </div>
        <button className="text-lg font-semibold bg-red-500 p-2 px-4 rounded-3xl text-white">
          Send
        </button>
      </div>
      <div
        onClick={() => {
          props.setConfirmRidePanel(true);
        }}
        className="flex border-2 active:border-black mb-2 rounded-xl w-full p-3  items-center justify-between"
      >
        <img className="h-10" src="/peace.png" alt="" />
        <div className="ml-2 w-1/2">
          <h4 className="font-medium text-base text-green-500">
            AskHelp <span>{/* <i className="ri-user-3-fill"></i>3 */}</span>
          </h4>
          <h5 className="font-medium text-sm">Ask Nearby People for help </h5>
          <p className="font-normal text-xs text-gray-600">
            Don't worry the helpers details will be provided to police also
          </p>
        </div>
        <button className="text-lg font-semibold bg-green-500 p-2 px-4 rounded-3xl border text-white border-green-700">
          Ask
        </button>
      </div>
    </div>
  );
};
export default VehiclePanel;
