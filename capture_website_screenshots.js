const path = require("path");
const fs = require("fs");
const { chromium } = require("playwright");

const outDir = path.join(__dirname, "output", "screenshots_viewport");
fs.mkdirSync(outDir, { recursive: true });

const pages = [
  ["home", "http://127.0.0.1:5000/"],
  ["dataset", "http://127.0.0.1:5000/dataset"],
  ["eda", "http://127.0.0.1:5000/eda"],
  ["visualizations", "http://127.0.0.1:5000/visualizations"],
  ["clustering", "http://127.0.0.1:5000/clustering"],
  ["cluster_visualization", "http://127.0.0.1:5000/cluster-visualization"],
  ["recommendations", "http://127.0.0.1:5000/recommendations"],
  ["dashboard", "http://127.0.0.1:5000/dashboard"],
  ["predict", "http://127.0.0.1:5000/predict"],
];

(async () => {
  const browser = await chromium.launch({
    headless: true,
    executablePath: "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
    args: ["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
  });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
  page.setDefaultTimeout(45000);

  for (const [name, url] of pages) {
    await page.goto(url, { waitUntil: "networkidle" });
    await page.waitForTimeout(3500);
    await page.screenshot({
      path: path.join(outDir, `${name}.png`),
      fullPage: false,
    });
    console.log(`${name}.png`);
  }

  await browser.close();
})();
