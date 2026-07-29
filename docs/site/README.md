# CudaRobotics docs site source

This directory contains a static documentation site that can be copied into the
existing `gh-pages` branch under `/docs/` without replacing the animated demo
gallery at the site root.

Preview locally:

```bash
python3 -m http.server 8080 --directory docs/site
```

Then open `http://localhost:8080/`.

Deploy manually to the legacy Pages branch:

```bash
git worktree add /tmp/cudarobotics-pages origin/gh-pages
rsync -a --delete docs/site/ /tmp/cudarobotics-pages/docs/
cd /tmp/cudarobotics-pages
git add docs
git commit -m "Publish docs site"
git push origin HEAD:gh-pages
```

Do not run a Pages workflow that replaces the whole `gh-pages` branch unless
the gallery assets have been migrated first.

For the v1 release, use the manual `v1-docs-deploy.yml` workflow. It checks
out the complete existing `gh-pages` tree, updates only `pages/docs/`, adds a
source-tag deployment manifest, and deploys the preserved complete tree. It
then re-fetches the public pages and uploads content-bound release evidence:

```bash
gh workflow run v1-docs-deploy.yml \
  --ref v1.0.0 -f tag=v1.0.0
```
