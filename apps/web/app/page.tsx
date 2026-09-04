import styles from './page.module.css';

const capabilities = [
  ['Chat', 'Reason, write, plan and solve in one workspace.'],
  ['Create', 'Build images, audio, video, documents and stories.'],
  ['Code', 'Design, edit, test and ship software with JagX Code.'],
  ['Research', 'Turn questions into structured, evidence-aware work.'],
];

export default function HomePage() {
  return (
    <main className={styles.page}>
      <nav className={styles.nav}><strong>JAGX AI</strong><span>Research</span><span>JagX Code</span><span>API</span><button>Sign in</button></nav>
      <section className={styles.hero}>
        <p className={styles.eyebrow}>AN INDEPENDENT AI WORKSPACE</p>
        <h1>One place to <span>think, create and build.</span></h1>
        <p className={styles.lede}>JagX AI brings conversation, creation, research and software development into one focused workspace.</p>
        <div className={styles.actions}><button className={styles.primary}>Start with JagX</button><button className={styles.secondary}>Explore JagX Code</button></div>
      </section>
      <section className={styles.grid}>{capabilities.map(([title, text]) => <article key={title}><p className={styles.cardLabel}>{title}</p><h2>{title}</h2><p>{text}</p></article>)}</section>
      <footer>JagX AI · Built for useful work · Privacy · Security · Documentation</footer>
    </main>
  );
}
