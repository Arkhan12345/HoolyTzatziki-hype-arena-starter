/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        sapphire: { DEFAULT: '#2859B5', light: '#4C7FE0' },
        lime: { DEFAULT: '#D7F13B', dim: '#8A9C14' },
        onyx: { DEFAULT: '#0A0A0A', 900: '#141414', 800: '#1E1E1E', 700: '#2A2A2A', 600: '#3A3A3A' },
        paper: '#FCFCFC',
        verdict: { true: '#3FB37F', false: '#E5484D', unsure: '#F5A524' },
        ink: { DEFAULT: '#FCFCFC', muted: '#A1A1A1', faint: '#6B6B6B' },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'Segoe UI', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'Consolas', 'monospace'],
      },
      keyframes: {
        pulseRing: { '0%,100%': { boxShadow: '0 0 0 0 rgba(215,241,59,0.55)' }, '50%': { boxShadow: '0 0 0 8px rgba(215,241,59,0)' } },
        stampIn: { '0%': { transform: 'scale(1.6) rotate(-8deg)', opacity: '0' }, '100%': { transform: 'scale(1) rotate(-4deg)', opacity: '1' } },
      },
      animation: {
        pulseRing: 'pulseRing 1.2s ease-in-out infinite',
        stampIn: 'stampIn 0.35s cubic-bezier(.2,1.4,.4,1) both',
      },
    },
  },
  plugins: [],
}
