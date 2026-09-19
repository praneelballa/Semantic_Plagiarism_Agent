/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        dark: {
          900: '#07090e',
          800: '#0f172a',
          700: '#1e293b',
          600: '#334155'
        },
        cyan: {
          DEFAULT: '#00f0ff',
          400: '#22d3ee',
          500: '#06b6d4'
        }
      }
    },
  },
  plugins: [],
}
