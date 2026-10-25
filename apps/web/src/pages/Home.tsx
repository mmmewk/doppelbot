import { Link } from "react-router";
import { TrainerBoard } from "../components/TrainerBoard.tsx";
import { displayName, useProfile } from "../profile.tsx";
import { TRAINERS } from "../trainers.ts";

/** "Let's get started", or "Play as <name>" once usernames are saved. */
function StartActions() {
  const { profile, openModal } = useProfile();
  return profile ? (
    <div className="cta-actions">
      <Link to="/play" className="button">
        Play as {displayName(profile)}
      </Link>
      <button className="button secondary" onClick={openModal}>
        Edit usernames
      </button>
    </div>
  ) : (
    <div className="cta-actions">
      <button className="button" onClick={openModal}>
        Let's get started
      </button>
    </div>
  );
}

export default function Home() {
  return (
    <main className="home">
      <section className="hero">
        <div className="hero-title">
          <h1>Chess trainers built from how you play.</h1>
          <StartActions />
        </div>
        <p className="lede">
          Doppelbot reads your Lichess and chess.com games and builds a model of your play. Then it
          gives you sparring partners that know your openings, your habits and your blind spots.
        </p>
        <p>
          It's built on <a href="https://www.maiachess.com">Maia</a>, the human-like chess model
          from the Maia Chess team at the University of Toronto's CSSLab, trained on games from{" "}
          <a href="https://lichess.org">Lichess</a>.
        </p>
      </section>

      <section className="trainers" aria-label="Trainers">
        {TRAINERS.map((trainer) => (
          <article key={trainer.name} className="trainer-card">
            <TrainerBoard
              id={`trainer-${trainer.name.toLowerCase()}`}
              line={trainer.line}
              label={trainer.boardLabel}
            />
            <div className="trainer-body">
              <header>
                <h2>{trainer.name}</h2>
                {!trainer.playable && <span className="badge">Coming soon</span>}
              </header>
              <p className="role">{trainer.role}</p>
              <p>{trainer.description}</p>
              <p className="time-control">
                <strong>Play it at:</strong> {trainer.timeControl}
              </p>
              {trainer.link && (
                <a href={trainer.link.href} target="_blank" rel="noreferrer">
                  {trainer.link.label} ↗
                </a>
              )}
            </div>
          </article>
        ))}
      </section>

      <section className="cta">
        <h2>Let's get started</h2>
        <p>
          Add your usernames and play Maia at your rating today. Your personal trainers come next.
        </p>
        <StartActions />
      </section>

      <footer className="credits">
        <p>
          Move predictions come from Maia-3, by the{" "}
          <a href="https://www.maiachess.com">Maia Chess</a> team at the{" "}
          <a href="https://csslab.cs.toronto.edu">University of Toronto CSSLab</a>: Monroe,
          Eilender, Chalmers, Tang and Anderson,{" "}
          <a href="https://github.com/CSSLab/maia3">
            "Chessformer: A Unified Architecture for Chess Modeling"
          </a>
          , ICLR 2026. Thanks to <a href="https://lichess.org">Lichess</a> for the open game data
          Maia is trained on, and for its public API.
        </p>
      </footer>
    </main>
  );
}
