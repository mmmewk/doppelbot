import { NavLink, Outlet } from "react-router";
import { ProfileModal } from "./components/ProfileModal.tsx";
import { ProfileProvider } from "./profile.tsx";

function App() {
  return (
    <ProfileProvider>
      <nav className="top-nav">
        <div className="top-nav-inner">
          <NavLink to="/" className="wordmark">
            doppelbot
          </NavLink>
          <div className="tabs">
            <NavLink to="/" end>
              Home
            </NavLink>
            <NavLink to="/play">Play</NavLink>
          </div>
        </div>
      </nav>
      <Outlet />
      <ProfileModal />
    </ProfileProvider>
  );
}

export default App;
