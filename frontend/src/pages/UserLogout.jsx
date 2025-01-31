import React from 'react'
import axios from 'axios'
import { useNavigate } from 'react-router-dom'

const UserLogout = () => {
  const token = localStorage.getItem('token');
  const navigate = useNavigate();

  axios.get(`${import.meta.env.VITE_BASE_URL}/users/logout`, {
    headers: {
      Authorization: `Bearer ${token}`
    }
  }).then((response) => {
    console.log(response);
    if (response.status === 200) {
      localStorage.removeItem('token');
      navigate('/login');
    }
  }).catch((error) => {
    console.log(error);
  });

  return (
    <div className='text-red-500 text-xl'>Now You are logged out </div>
  )
}

export default UserLogout