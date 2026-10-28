/** @type {import('tailwindcss').Config} */
// Hype Arena design tokens: see DESIGN.md. Components use these names, never raw hex.
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        void: '#05070D',
        surface: { 1: '#0A0F1C', 2: '#0E1526', 3: '#142036' },
        line: { DEFAULT: '#1B2842', strong: '#2A3D61' },
        ink: { DEFAULT: '#E6EEFB', muted: '#9AABC8', faint: '#7183A3' },
        cyan: { DEFAULT: '#22D3EE', ice: '#A5F3FC', deep: '#0E7490' },
        verdict: { true: '#B8F23A', false: '#EF6B73', unsure: '#F5B83D' },
      },
      fontFamily: {
        display: ['Syncopate', 'IBM Plex Sans', 'sans-serif'],
        sans: ['IBM Plex Sans', 'system-ui', 'Segoe UI', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'Consolas', 'monospace'],
      },
      fontSize: {
        '2xs': ['0.6875rem', { lineHeight: '1rem' }],
      },
      boxShadow: {
        glow: '0 0 0 1px rgba(34,211,238,0.35), 0 0 24px -4px rgba(34,211,238,0.45)',
        'glow-true': '0 0 0 1px rgba(184,242,58,0.4), 0 0 28px -6px rgba(184,242,58,0.5)',
        panel: '0 1px 0 0 rgba(165,243,252,0.04) inset, 0 24px 48px -24px rgba(0,0,0,0.8)',
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
