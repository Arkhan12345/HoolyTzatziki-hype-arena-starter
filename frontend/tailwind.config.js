/** @type {import('tailwindcss').Config} */
// Hype Arena design tokens: CSS variables per theme in src/index.css, see DESIGN.md. Components use these names, never raw hex.
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: ['selector', ':root:not([data-theme="light"])'],
  theme: {
    extend: {
      colors: {
        void: 'rgb(var(--void) / <alpha-value>)',
        surface: { 1: 'rgb(var(--surface-1) / <alpha-value>)', 2: 'rgb(var(--surface-2) / <alpha-value>)', 3: 'rgb(var(--surface-3) / <alpha-value>)' },
        line: { DEFAULT: 'rgb(var(--line) / <alpha-value>)', strong: 'rgb(var(--line-strong) / <alpha-value>)' },
        ink: { DEFAULT: 'rgb(var(--ink) / <alpha-value>)', muted: 'rgb(var(--ink-muted) / <alpha-value>)', faint: 'rgb(var(--ink-faint) / <alpha-value>)' },
        cyan: { DEFAULT: 'rgb(var(--cyan) / <alpha-value>)', ice: 'rgb(var(--cyan-ice) / <alpha-value>)', deep: 'rgb(var(--cyan-deep) / <alpha-value>)' },
        'on-accent': 'rgb(var(--on-accent) / <alpha-value>)',
        verdict: { true: 'rgb(var(--v-true) / <alpha-value>)', false: 'rgb(var(--v-false) / <alpha-value>)', unsure: 'rgb(var(--v-unsure) / <alpha-value>)' },
      },
      fontFamily: {
        display: ['IBM Plex Sans', 'system-ui', 'sans-serif'],
        sans: ['IBM Plex Sans', 'system-ui', 'Segoe UI', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'Consolas', 'monospace'],
      },
      fontSize: {
        '2xs': ['0.6875rem', { lineHeight: '1rem' }],
      },
      boxShadow: {
        glow: '0 0 0 1px rgb(var(--cyan) / 0.35), 0 0 24px -4px rgb(var(--cyan) / 0.45)',
        'glow-true': '0 0 0 1px rgb(var(--v-true) / 0.4), 0 0 28px -6px rgb(var(--v-true) / 0.5)',
        panel: '0 1px 0 0 rgb(var(--cyan-ice) / 0.04) inset, 0 24px 48px -24px rgb(var(--shadow) / 0.35)',
      },
      keyframes: {
        drift: { '0%,100%': { transform: 'translate3d(0,0,0) scale(1)' }, '50%': { transform: 'translate3d(4%,-3%,0) scale(1.08)' } },
        gridPan: { from: { backgroundPosition: '0 0' }, to: { backgroundPosition: '0 64px' } },
        scan: { from: { transform: 'translateY(-100%)' }, to: { transform: 'translateY(100vh)' } },
        blink: { '0%,49%': { opacity: '1' }, '50%,100%': { opacity: '0' } },
        flow: { to: { strokeDashoffset: '-24' } },
        shimmer: { from: { backgroundPosition: '-200% 0' }, to: { backgroundPosition: '200% 0' } },
        spinSlow: { to: { transform: 'rotate(360deg)' } },
        float: { '0%,100%': { transform: 'translateY(0) rotate(0deg)' }, '50%': { transform: 'translateY(-14px) rotate(8deg)' } },
      },
      animation: {
        drift: 'drift 26s ease-in-out infinite',
        'drift-slow': 'drift 38s ease-in-out infinite reverse',
        gridPan: 'gridPan 6s linear infinite',
        scan: 'scan 9s linear infinite',
        blink: 'blink 1s step-end infinite',
        flow: 'flow 0.9s linear infinite',
        shimmer: 'shimmer 2.2s linear infinite',
        spinSlow: 'spinSlow 18s linear infinite',
        float: 'float 14s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}
