import React from 'react'
import { createRoot } from 'react-dom/client'
import { HashRouter } from 'react-router-dom'
import '@fontsource-variable/nunito'
import '@fontsource-variable/fredoka'
import App from './App'
import { HardwareProvider, applyTheme, getTheme } from './api'
import './styles.css'

applyTheme(getTheme())

createRoot(document.getElementById('root')).render(
  <HashRouter>
    <HardwareProvider>
      <App />
    </HardwareProvider>
  </HashRouter>,
)
