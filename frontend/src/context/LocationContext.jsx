import React, { createContext, useState } from "react";
export const ToLocationContext = createContext();
const ToLocationProvider = ({ children }) => {
  const [toLocation, setToLocation] = useState([0,0]);
  return (
    <div>
      <ToLocationContext.Provider value={{ toLocation, setToLocation }}>
        {children}
      </ToLocationContext.Provider>
    </div>
  );
};
export default ToLocationProvider;
