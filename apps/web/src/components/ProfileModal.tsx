import { useEffect, useRef, useState, type FormEvent } from "react";
import { useNavigate } from "react-router";
import { displayName, useProfile } from "../profile.tsx";
import { UserNotFoundError, lookupChessCom, lookupLichess } from "../ratings.ts";

/** Asks for chess.com and Lichess usernames, saves them locally, then goes to Play. */
export function ProfileModal() {
  const { profile, saveProfile, modalOpen, closeModal } = useProfile();
  const navigate = useNavigate();
  const dialog = useRef<HTMLDialogElement>(null);
  const [name, setName] = useState("");
  const [chesscom, setChesscom] = useState("");
  const [lichess, setLichess] = useState("");
  const [errors, setErrors] = useState<{ chesscom?: string; lichess?: string; form?: string }>({});
  const [checking, setChecking] = useState(false);

  useEffect(() => {
    if (modalOpen) {
      setName(profile?.name ?? "");
      setChesscom(profile?.chesscom ?? "");
      setLichess(profile?.lichess ?? "");
      setErrors({});
      dialog.current?.showModal();
    } else {
      dialog.current?.close();
    }
  }, [modalOpen, profile]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    const names = { chesscom: chesscom.trim(), lichess: lichess.trim() };
    if (!names.chesscom && !names.lichess) {
      setErrors({ form: "Enter at least one username." });
      return;
    }
    setChecking(true);
    // Only a definite "no such user" blocks saving; if a site is unreachable, save anyway.
    const check = async (name: string, lookup: (name: string) => Promise<unknown>) => {
      if (!name) return undefined;
      try {
        await lookup(name);
        return undefined;
      } catch (error) {
        return error instanceof UserNotFoundError ? error.message : undefined;
      }
    };
    const [chesscomError, lichessError] = await Promise.all([
      check(names.chesscom, lookupChessCom),
      check(names.lichess, lookupLichess),
    ]);
    setChecking(false);
    if (chesscomError || lichessError) {
      setErrors({ chesscom: chesscomError, lichess: lichessError });
      return;
    }
    saveProfile({
      name: name.trim() || undefined,
      chesscom: names.chesscom || undefined,
      lichess: names.lichess || undefined,
      savedAt: new Date().toISOString(),
    });
    closeModal();
    void navigate("/play");
  }

  return (
    <dialog ref={dialog} className="modal" onClose={closeModal} aria-labelledby="profile-title">
      <form onSubmit={submit} noValidate>
        <h2 id="profile-title">Who are you on the board?</h2>
        <p className="muted">
          Add one or both usernames. Your trainers will learn from your games on these accounts.
        </p>
        <label>
          chess.com username
          <input
            value={chesscom}
            onChange={(event) => setChesscom(event.target.value)}
            autoComplete="off"
            spellCheck={false}
            aria-invalid={Boolean(errors.chesscom)}
          />
          {errors.chesscom && <span className="field-error">{errors.chesscom}</span>}
        </label>
        <label>
          Lichess username
          <input
            value={lichess}
            onChange={(event) => setLichess(event.target.value)}
            autoComplete="off"
            spellCheck={false}
            aria-invalid={Boolean(errors.lichess)}
          />
          {errors.lichess && <span className="field-error">{errors.lichess}</span>}
        </label>
        <label>
          <span>
            Your name <span className="muted">(optional)</span>
          </span>
          <input
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder={displayName({ lichess: lichess.trim(), chesscom: chesscom.trim() })}
            autoComplete="given-name"
          />
        </label>
        {errors.form && <p className="field-error">{errors.form}</p>}
        <p className="fine-print">
          Saved in this browser only. Nothing is uploaded, and no model is trained yet.
        </p>
        <div className="modal-actions">
          <button type="button" className="button secondary" onClick={closeModal}>
            Cancel
          </button>
          <button type="submit" className="button" disabled={checking}>
            {checking ? "Checking…" : "Save and play"}
          </button>
        </div>
      </form>
    </dialog>
  );
}
