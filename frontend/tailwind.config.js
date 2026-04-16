/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'space-dark': '#0a0a0f',
        'space-light': '#1a1730',
        'neon-blue': '#00d2ff',
        'neon-purple': '#a200ff',
        'glass-bg': 'rgba(255, 255, 255, 0.05)',
        'glass-border': 'rgba(255, 255, 255, 0.1)',
      },
      fontFamily: {
        sans: ['"Fira Sans"', 'sans-serif'],
        mono: ['"Fira Code"', 'monospace'],
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'cyber-gradient': 'linear-gradient(135deg, #0a0a0f 0%, #0f0c29 40%, #302b63 80%, #24243e 100%)',
      },
    },
  },
  plugins: [],
}
