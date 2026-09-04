import styles from './code.module.css';

const files = ['src/app.ts', 'src/index.ts', 'tests/app.test.ts', 'package.json'];

export default function CodePage() {
  return <main className={styles.shell}>
    <aside><strong>JAGX CODE</strong><p className={styles.muted}>WORKSPACE</p>{files.map((file, i) => <div className={i === 0 ? styles.active : styles.file} key={file}>{file}</div>)}</aside>
    <section className={styles.editor}>
      <header><span>app.ts</span><button>Run</button></header>
      <pre><code>{`import { createJagXApp } from '@jagx/runtime';\n\nconst app = createJagXApp({\n  workspace: 'default',\n  tools: ['terminal', 'research'],\n});\n\napp.start();`}</code></pre>
    </section>
    <section className={styles.ai}><p className={styles.muted}>JAGX ASSISTANT</p><h2>What should we build?</h2><p>Explain code, generate files, debug failures, write tests or prepare a patch.</p><div className={styles.actions}><button>Explain</button><button>Generate tests</button><button>Find bug</button><button>Refactor</button></div><div className={styles.terminal}><p>$ jagx test</p><p>Waiting for workspace execution…</p></div></section>
  </main>;
}
