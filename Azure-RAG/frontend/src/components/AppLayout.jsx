import { NavLink, Outlet } from 'react-router-dom'

const navItems = [
  { to: '/', label: 'Home', end: true },
  { to: '/wiki', label: 'Create Wiki LLM' },
  { to: '/chat', label: 'RAG Chat' },
]

export default function AppLayout() {
  return (
    <div className="app-shell">
      <nav className="navbar" aria-label="Main navigation">
        <NavLink className="brand" to="/">
          GovTech Knowledge AI
        </NavLink>
        <div className="nav-links">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
            >
              {item.label}
            </NavLink>
          ))}
        </div>
      </nav>

      <Outlet />
    </div>
  )
}
