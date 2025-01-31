import React from "react";
import { Link } from "react-router-dom";
const Land = () => {
  return (
    <div>
      <div className="bg-cover bg-center bg-[url('/drive.jpeg')] h-screen pt-8 flex justify-between flex-col w-full">
        <img
          className="w-40 ml-8 filter"
          src="/logo.jpg"
          alt=""
        />
        <div className="bg-white pb-8 py-4 px-4">
          <h2 className="text-[30px] font-semibold">Get Started with SafeThrough</h2>
          <Link
            to="/login"
            className="flex items-center justify-center w-full bg-black text-white py-3 rounded-lg mt-5"
          >
            Continue
          </Link>
        </div>
      </div>
    </div>
  );
};
export default Land;
