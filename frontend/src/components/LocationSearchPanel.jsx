import React, { useEffect, useState } from "react";
import axios from "axios";

const LocationSearchPanel = (props) => {
  const [input, setInput] = useState("");
  const [suggestions, setSuggestions] = useState([]);

  const fetchSuggestions = async (input) => {
    try {
      const response = await axios.get(
        `${import.meta.env.VITE_BASE_URL}/maps/get-suggestions?input=${input}`,{
        headers : {
          Authorization: `Bearer ${localStorage.getItem("token")}`,
      }}
      );
      if (response.data) {
        setSuggestions(response.data);
      }
    } catch (error) {
      console.error("Error fetching suggestions:", error);
    }
  };

  // const handleInputChange = (e) => {
  //   const value = e.target.value;
  //   setInput(value);
  //   if (value.length > 2) {
  //     fetchSuggestions(value);
  //   } else {
  //     setSuggestions([]);
  //   }
  // };
  useEffect(() => {
    const value = props.isPickup ? props.pickup : props.destination;
    if (value.length > 2) {
      fetchSuggestions(value);
    } else {
      setSuggestions([]);
    }
  }, [props.pickup, props.destination]);
  const handleSuggestionClick = (suggestion) => {
    props.setLocation(suggestion.description);
  };

  return (
    <div>
      {suggestions.map((suggestion, idx) => (
        <div
          key={idx}
          onClick={() => handleSuggestionClick(suggestion)}
          className="flex gap-4 border-2 p-3 border-gray-50 active:border-black rounded-xl items-center my-2 justify-start"
        >
          <h2 className="bg-[#eee] h-8 flex items-center justify-center w-12 rounded-full">
            <i className="ri-map-pin-fill"></i>
          </h2>
          <h4 className="font-medium">{suggestion.description}</h4>
        </div>
      ))}
    </div>
  );
};

export default LocationSearchPanel;