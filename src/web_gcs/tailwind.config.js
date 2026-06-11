/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        'tech-bg': '#0a0e17',
        'tech-panel': '#111827',
        'tech-border': '#1f2937',
        'tech-blue': '#3b82f6',
        'tech-green': '#10b981',
        'tech-red': '#ef4444',
        'tech-yellow': '#f59e0b',
        'tech-cyan': '#06b6d4',
      },
      fontFamily: {
        'tech': ['Inter', 'system-ui', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
