# TOOLDEX — project website

Static site for **TOOLDEX: An Ego-Exo Human Demonstration Dataset for Creative Physical Tool Use**.

Served by GitHub Pages at the repository's Pages URL. There is no build step — the site is plain
HTML, CSS and JS, deployed on every push to `main` by the workflow in `.github/workflows/`.

## Layout

| Path | Contents |
|---|---|
| `index.html` | main project page |
| `dataset-visualizer/` | filterable task browser; cards toggle between the original camera view and the 4DGS novel view |
| `8cam-vs-10cam/` | wipe-slider comparison of the 8-camera and 10-camera 4DGS reconstructions |
| `static/` | CSS and JS, including the novel-view comparison sliders |
| `figures/`, `videos/`, `stats/`, `assets/` | page media and precomputed statistics |

## Credits

The novel-view comparison sliders are adapted from
[Ref-NeRF](https://dorverbin.github.io/refnerf/)'s `video_comparison.js`, generalised from two panes
to N panes. The page structure follows the
[DROID](https://droid-dataset.github.io/) dataset site.

## Note

This repository is kept anonymous while the associated paper is under double-blind review.
