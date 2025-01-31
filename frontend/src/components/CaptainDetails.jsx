import React, { useContext } from "react";
import { CaptainDataContext } from "../context/CaptainContext";

const CaptainDetails = () => {
  const { captain } = useContext(CaptainDataContext);
  // console.log(captain, "In details");
  
  return (
    <div>
      <div className="flex items-center justify-between">
        <div className="flex items-center justify-start gap-3">
          <img
            className="h-10 w-10 rounded-full object-cover"
            src="/peace.png"
            alt=""
          />
          <h4 className="text-lg font-medium capitalize">
            {captain?.fullName.firstName + " " + captain?.fullName.lastName}
          </h4>
        </div>
        <div>
          <h4 className="text-xl font-semibold">5 assists</h4>
          <p className="text-sm text-gray-600">Done</p>
        </div>
      </div>
      <div className="flex p-3 mt-8 bg-gray-100 rounded-xl justify-center gap-5 items-start">
        <div className="text-center">
          <i className="text-3xl mb-2 font-thin ri-timer-2-line"></i>
          <h5 className="text-lg font-medium">4.2</h5>
          <p className="text-sm text-gray-600">Hours Dedicated</p>
        </div>
        <div className="text-center">
          <img src="/pointed-star.png" className="w-9"></img>
          <h5 className="text-lg font-medium">4.6</h5>
          <p className="text-sm text-gray-600">Rating</p>
        </div>
        <div className="text-center">
          <i className="text-3xl mb-2 font-thin ri-booklet-line"></i>
          <h5 className="text-lg font-medium">250</h5>
          <p className="text-sm text-gray-600">Earned as appreciation</p>
        </div>
      </div>
    </div>
  );
};

export default CaptainDetails;
