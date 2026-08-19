// Assemble the self-contained page: inline sim.js into the template.
const fs = require('fs');
const path = require('path');
const tpl = fs.readFileSync(path.join(__dirname, 'index.template.html'), 'utf8');
const sim = fs.readFileSync(path.join(__dirname, 'sim.js'), 'utf8');
if (!tpl.includes('/* INJECT:SIM */')) throw new Error('inject marker missing');
const out = tpl.replace('/* INJECT:SIM */', () => sim);
fs.mkdirSync(path.join(__dirname, 'dist'), { recursive: true });
fs.writeFileSync(path.join(__dirname, 'dist', 'crease-lab.html'), out);
console.log('wrote dist/crease-lab.html', (out.length / 1024).toFixed(0) + 'kB');
