# AI Assessment frontend

React, Vite, Tailwind CSS, React Router, and Axios frontend for the three assessment tasks and additional hybrid crack progression analysis.

```powershell
npm ci
npm run dev
npm test
npm run build
```

The development server uses port 5173. The API URL defaults to http://localhost:8000/api; see .env.example for an optional override. Start the backend services separately using the [project README](../README.md).

Routes: / (dashboard), /inspection, /image-comparison, /news-classification, /features, /similarity, /hybrid-analysis, and /system-status.

Keep package-lock.json in source control. node_modules/, dist/, and local environment files are ignored. tests/e2e_real_verification.js is a separate live-service verification utility; npm test runs the Node test suite.
