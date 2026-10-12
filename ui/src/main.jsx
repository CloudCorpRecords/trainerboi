import React from 'react'
import { createRoot } from 'react-dom/client'
import { HashRouter } from 'react-router-dom'
import App from './App'
import { HardwareProvider } from './api'
import './styles.css'

createRoot(document.getElementById('root')).render(
  <HashRouter>
    <HardwareProvider>
      <App />
    </HardwareProvider>
  </HashRouter>,
)
