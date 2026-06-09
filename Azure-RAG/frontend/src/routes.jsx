import { Routes, Route, Navigate } from 'react-router-dom'
import AppLayout from './components/AppLayout'
import HomePage from './pages/HomePage'
import ChatPage from './pages/ChatPage'
import BuildPage from './pages/BuildPage'

export default function AppRoutes() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route path="/" element={<HomePage />} />
        <Route path="/wiki" element={<BuildPage />} />
        <Route path="/chat" element={<ChatPage />} />
      </Route>
      <Route path="/build" element={<Navigate to="/wiki" replace />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
