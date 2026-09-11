import { FileCode2 } from 'lucide-react';

// Placeholder until the real X handle is chosen.
const X_URL = 'https://x.com';

function XLogo() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
      <path
        fill="currentColor"
        d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"
      />
    </svg>
  );
}

export function CommunitySection() {
  return (
    <section className="community-section" aria-labelledby="community-title">
      <h2 id="community-title" className="workspace-eyebrow">
        Community
      </h2>
      <div className="community-grid">
        <a
          className="workspace-panel community-card"
          href={X_URL}
          target="_blank"
          rel="noopener noreferrer"
        >
          <span className="community-icon">
            <XLogo />
          </span>
          <span>
            <strong>Follow on X</strong>
            <small>Flight logs, brain experiments and updates.</small>
          </span>
          <span className="community-arrow" aria-hidden="true">
            →
          </span>
        </a>
        <div className="workspace-panel community-card">
          <span className="community-icon">
            <FileCode2 size={18} />
          </span>
          <span>
            <strong>Contract</strong>
            <small>The contract address will be published here.</small>
          </span>
          <span className="source-pill">COMING SOON</span>
        </div>
      </div>
    </section>
  );
}
