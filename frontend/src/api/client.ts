import axios from "axios";

// Empty base means requests go through the Vite dev proxy (/api -> backend).
// Set VITE_API_BASE to point at a deployed backend.
const API_BASE = (import.meta.env.VITE_API_BASE as string) || "";
const api = axios.create({ baseURL: API_BASE });

export const getOverview = () => api.get("/api/demo/overview").then((r) => r.data);
export const getCaar = () => api.get("/api/demo/caar").then((r) => r.data);
export const getCorrelation = () => api.get("/api/demo/correlation").then((r) => r.data);
export const getDecay = () => api.get("/api/demo/decay").then((r) => r.data);
export const scoreText = (text: string) =>
  api.post("/api/sentiment/score", { text }).then((r) => r.data);
export const extractTickers = (text: string) =>
  api.post("/api/tickers/extract", { text }).then((r) => r.data);
