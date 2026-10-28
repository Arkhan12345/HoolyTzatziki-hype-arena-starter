// Copies ../pitch/PITCH.md (P1's file) to public/pitch.md so /pitch always shows the latest version.
// Run automatically before `npm run dev` and `npm run build`. A plain copy, no symlink (Windows-safe).
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))
const src = path.resolve(here, '../../pitch/PITCH.md')
const dst = path.resolve(here, '../public/pitch.md')

if (fs.existsSync(src)) {
  fs.copyFileSync(src, dst)
  console.log(`copy-pitch: ${path.relative(process.cwd(), src)} -> public/pitch.md`)
} else {
  if (fs.existsSync(dst)) fs.rmSync(dst)
  console.log('copy-pitch: pitch/PITCH.md not found yet; /pitch shows its placeholder deck')
}
