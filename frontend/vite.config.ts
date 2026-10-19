import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `base: "./"` keeps the build relocatable. In development the API runs separately
// (`miraichem serve`) and is proxied, so the browser sees one origin.
export default defineConfig({
  plugins: [react()],
  base: "./",
  server: {
    port: 5173,
    host: "127.0.0.1",
    proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: false } },
  },
  test: { environment: "node" },
});
