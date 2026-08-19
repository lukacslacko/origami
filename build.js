// Assemble the self-contained page: inline sim.js into the template.
const fs = require('fs');
const path = require('path');
const tpl = fs.readFileSync(path.join(__dirname, 'index.template.html'), 'utf8');
const sim = fs.readFileSync(path.join(__dirname, 'sim.js'), 'utf8');
const gpu = fs.readFileSync(path.join(__dirname, 'gpu.js'), 'utf8');
if (!tpl.includes('/* INJECT:SIM */')) throw new Error('sim inject marker missing');
if (!tpl.includes('/* INJECT:GPU */')) throw new Error('gpu inject marker missing');
const out = tpl.replace('/* INJECT:SIM */', () => sim).replace('/* INJECT:GPU */', () => gpu);
fs.mkdirSync(path.join(__dirname, 'dist'), { recursive: true });
fs.writeFileSync(path.join(__dirname, 'dist', 'crease-lab.html'), out);
// standalone copy for GitHub Pages
fs.mkdirSync(path.join(__dirname, 'docs'), { recursive: true });
fs.writeFileSync(path.join(__dirname, 'docs', 'index.html'),
  '<!doctype html>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width,initial-scale=1">\n' + out);
console.log('wrote dist/crease-lab.html + docs/index.html', (out.length / 1024).toFixed(0) + 'kB');
