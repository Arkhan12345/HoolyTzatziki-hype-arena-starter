// Copies shared repo files the UI reads at runtime into public/ (plain copies, no symlinks, Windows-safe):
//   ../pitch/PITCH.md     -> public/pitch.md            (P1's deck for /pitch)
//   ../data/lexicon.json  -> public/data/lexicon.json   (entity + attribute lookup)
//   ../data/practice.csv  -> public/data/practice.csv   (gold rows for the explorer and value timelines)
// Runs before `npm run dev` and `npm run build`. Missing sources are skipped; the UI shows its empty states.
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))
const repo = path.resolve(here, '../..')
const pub = path.resolve(here, '../public')

const copies = [
  ['pitch/PITCH.md', 'pitch.md'],
  ['data/lexicon.json', 'data/lexicon.json'],
  ['data/practice.csv', 'data/practice.csv'],
]

for (const [from, to] of copies) {
  const src = path.join(repo, from)
  const dst = path.join(pub, to)
  if (fs.existsSync(src)) {
    fs.mkdirSync(path.dirname(dst), { recursive: true })
    fs.copyFileSync(src, dst)
    console.log(`sync-assets: ${from} -> public/${to}`)
  } else {
    if (fs.existsSync(dst)) fs.rmSync(dst)
    console.log(`sync-assets: ${from} not found, skipped`)
  }
}
