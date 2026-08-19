// Assemble the self-contained page: inline sim.js into the template.
const fs = require('fs');
const path = require('path');
const tpl = fs.readFileSync(path.join(__dirname, 'index.template.html'), 'utf8');
const inject = {
  'SIM': 'sim.js', 'GPU': 'gpu.js', 'MESH2': 'mesh2.js', 'SIM2': 'sim2.js',
};
let out = tpl;
for (const [tag, file] of Object.entries(inject)) {
  const marker = `/* INJECT:${tag} */`;
  if (!out.includes(marker)) throw new Error(`${tag} inject marker missing`);
  const src = fs.readFileSync(path.join(__dirname, file), 'utf8');
  out = out.replace(marker, () => src);
}
fs.mkdirSync(path.join(__dirname, 'dist'), { recursive: true });
fs.writeFileSync(path.join(__dirname, 'dist', 'crease-lab.html'), out);
// standalone copy for GitHub Pages
fs.mkdirSync(path.join(__dirname, 'docs'), { recursive: true });
fs.writeFileSync(path.join(__dirname, 'docs', 'index.html'),
  '<!doctype html>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width,initial-scale=1">\n' + out);
console.log('wrote dist/crease-lab.html + docs/index.html', (out.length / 1024).toFixed(0) + 'kB');
