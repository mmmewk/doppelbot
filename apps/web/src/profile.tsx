import { createContext, useContext, useState, type ReactNode } from "react";

/** The user's usernames. Kept in the browser only; nothing is uploaded or trained yet. */
export type Profile = { name?: string; chesscom?: string; lichess?: string; savedAt: string };

const STORAGE_KEY = "doppelbot.profile";

function readProfile(): Profile | undefined {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? (JSON.parse(raw) as Profile) : undefined;
    return parsed?.chesscom || parsed?.lichess ? parsed : undefined;
  } catch {
    return undefined;
  }
}

function writeProfile(profile: Profile) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(profile));
  } catch {
    // Storage unavailable (private window, blocked site data): the profile lives in memory only.
  }
}

/** The name to greet the user by: their chosen name, else a username (Lichess wins when both are set). */
export function displayName(profile: Partial<Profile>): string {
  return profile.name || profile.lichess || profile.chesscom || "";
}

type ProfileContext = {
  profile: Profile | undefined;
  saveProfile: (profile: Profile) => void;
  modalOpen: boolean;
  openModal: () => void;
  closeModal: () => void;
};

const Context = createContext<ProfileContext | undefined>(undefined);

export function ProfileProvider({ children }: { children: ReactNode }) {
  const [profile, setProfile] = useState(readProfile);
  const [modalOpen, setModalOpen] = useState(false);
  function saveProfile(next: Profile) {
    writeProfile(next);
    setProfile(next);
  }
  return (
    <Context
      value={{
        profile,
        saveProfile,
        modalOpen,
        openModal: () => setModalOpen(true),
        closeModal: () => setModalOpen(false),
      }}
    >
      {children}
    </Context>
  );
}

export function useProfile(): ProfileContext {
  const context = useContext(Context);
  if (!context) throw new Error("useProfile must be used inside ProfileProvider");
  return context;
}
