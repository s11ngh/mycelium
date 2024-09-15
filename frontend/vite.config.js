import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  jsxRuntime: "classic", // Add this line

  plugins: [react()],
});
