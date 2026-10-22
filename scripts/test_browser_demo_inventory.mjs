import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile, readdir} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';

test('every executable browser demo is inventoried and offers a real comparison', async () => {
  const root = fileURLToPath(new URL('../docs/site/', import.meta.url));
  const inventory = JSON.parse(await readFile(path.join(root, 'demos.json'), 'utf8'));
  const discovered = [];
  for (const filename of await readdir(root, {recursive: true})) {
    if (!filename.endsWith('.html')) continue;
    const html = await readFile(path.join(root, filename), 'utf8');
    if (/<script[^>]*type="module"/.test(html)) discovered.push(filename.replaceAll('\\', '/'));
  }
  assert.deepEqual(discovered.sort(), inventory.demos.map(demo => demo.page).sort());
  for (const demo of inventory.demos) {
    const html = await readFile(path.join(root, demo.page), 'utf8');
    for (const identifier of ['compare', 'export-comparison', 'cpu-median', 'gpu-median', 'accuracy', 'speedup']) {
      assert.ok(html.includes(`id="${identifier}"`), `${demo.id}: missing comparison UI ${identifier}`);
    }
    assert.ok((await readFile(path.join(root, demo.comparison), 'utf8')).includes('runComparison'));
    assert.ok(demo.scope.length > 20);
  }
  assert.equal(inventory.recordedGallery.liveBenchmark, false);
});

test('the landing page has three direct demo entries and preserves access to research', async () => {
  const html = await readFile(new URL('../docs/landing/index.html', import.meta.url), 'utf8');
  const inventory = JSON.parse(await readFile(new URL('../docs/site/demos.json', import.meta.url), 'utf8'));
  const cards = [...html.matchAll(/<a class="demo[^\"]*" href="([^"]+)"/g)].map(match => match[1]);
  assert.equal(cards.length, 3);
  assert.deepEqual(cards.sort(), inventory.demos.map(demo => `docs/${path.posix.dirname(demo.page)}/`).sort());
  assert.ok(html.includes('href="research.html"'));
  assert.ok(html.includes('href="docs/install.html"'));
  assert.ok(html.includes('lang="ja"'));
  assert.ok(html.includes('ネイティブCUDAの性能測定ではありません'));
  assert.ok(!html.includes('<script'));
  assert.ok(inventory.recordedGallery.url.endsWith('/research.html'));
  const stylesheet = await readFile(new URL('../docs/landing/landing.css', import.meta.url), 'utf8');
  assert.ok(stylesheet.includes('prefers-reduced-motion'));
  assert.ok(stylesheet.includes(':focus-visible'));
});
