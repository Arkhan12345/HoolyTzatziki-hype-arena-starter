# Hype Arena design system

Chosen with the `ui-ux-pro-max` skill. Its `--design-system` query came back off-target (brutalism, a purple light palette), so we used its narrower, verified match: **Dark Mode (OLED)**, best for coding platforms and low-light use. That match calls for deep blacks and midnight-blue surfaces, neon accents used sparingly, minimal glow, text contrast of 7:1 or more, and visible focus. We tuned it to the brief: an intelligence terminal with an AI-lab feel, not a template SaaS page.

## Themes

There are two themes, dark (the default) and light. The toggle sits in the sidebar and the mobile top bar. The choice is saved in localStorage; without one, the system preference decides. An inline script in index.html sets  before first paint, so the page never flashes the wrong theme. Every token is a CSS variable in , holding an RGB triplet, so Tailwind can add alpha. SVG and Recharts colours come from  (), which reads the same variables, so there is one source of truth. In light mode the verdict colours are darker (lime , red , amber ) to keep text contrast at 4.5:1 or more on white.

## Tokens (, dark values shown)

| Role | Token | Value | Use |
| --- | --- | --- | --- |
| Page | `void` | `#05070D` | body, terminal wells |
| Surfaces | `surface-1/2/3` | `#0A0F1C` `#0E1526` `#142036` | panels, inputs, tracks |
| Lines | `line`, `line-strong` | `#1B2842` `#2A3D61` | hairlines, idle diagram nodes |
| Text | `ink`, `ink-muted`, `ink-faint` | `#E6EEFB` `#9AABC8` `#7183A3` | body ≥ 7:1, secondary, captions (≥ 4.5:1) |
| Interaction | `cyan`, `cyan-ice`, `cyan-deep` | `#22D3EE` `#A5F3FC` `#0E7490` | primary actions, focus, links, running state |
| True | `verdict-true` | `#B8F23A` acid lime | verified / done |
| False | `verdict-false` | `#EF6B73` muted red | contradiction |
| Unsure | `verdict-unsure` | `#F5B83D` amber | uncertainty, degraded stages, warnings |

Claim types use a fixed categorical order, checked with the dataviz palette validator on `surface-1`: fact `#3C8DE0`, compare `#C9608F`, change `#0FA3B8`, at_event `#8E6FE0`, relation `#6E8F1F`. The validator passes every check except one CVD warning, which is allowed only with secondary encoding, so a type is always shown with its name.

A verdict is never shown by colour alone. `VerdictBadge` always carries an icon and the word.

## Type

- **Syncopate**: the wordmark and display accents only. It is wide and futuristic.
- **IBM Plex Sans**: interface and body text. It reads technically and stays readable at density.
- **JetBrains Mono**: ids, dates, confidences, stage logs, forms and eyebrows, with tabular numerals.

## Motion

framer-motion is the only animation library. GSAP is not needed. Easing is `cubic-bezier(.16,1,.3,1)` on arrival. Exits run at about 60% of the enter time.

- **Atmosphere** (`fx/Backdrop`): drifting light fields, a perspective grid floor that pans, three wireframe shapes, a cursor light, noise and a slow scan line. All of it is fixed, `pointer-events: none`, and animated with transforms only.
- **Entrance**: the wordmark letters spring in staggered. Page headers reveal eyebrow, title and lead in a short stagger. Routes cross-fade.
- **Boot sequence**: shown once per session. Every line is a real check (API health, lexicon, practice data). Any key skips it.
- **Interaction**: magnetic primary buttons, a page-wide cursor light (`GlobalGlow`: one listener lights every panel relative to the cursor, so it follows the cursor across panels and never sticks to the last one), a conic border beam on key panels, spring tabs and expanders, and counting numbers.
- **Data**: the confidence dial sweeps in, distribution and histogram bars grow, terminal lines type in, and evidence-graph edges flow.
- **Pipeline honesty**: the API does not report progress mid-run, so during a request every stage shows an indeterminate "running" state. When the answer lands, the measured stage timings replay in order, and each stage ends as done or degraded (amber), based on its `stage_details` status.

`prefers-reduced-motion` turns off the backdrop movement, cursor light, scan line, boot sequence and replay, and every element renders in its final state.

## Confidence semantics

- The contract's confidence range is 0.5–1.0. The gauge draws 0–0.5 as a dim "below the floor" band.
- `unsure` has no confidence by contract. It shows a dashed amber ring and "—". It is never shown as a low-confidence false.
- Model confidence, evidence extracted, and what decided the verdict (classifier or baseline prior) are three separate readouts.
- While the classifier is the baseline, the UI says that its 0.83 is the share of true rows in practice. It is a prior, not evidence.
