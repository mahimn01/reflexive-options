# Zenodo DOI integration walkthrough

## What this gets you

A permanent, citable DOI for every tagged GitHub release of
`mahimn01/reflexive-options`, plus a "Cite this repository" sidebar
button on the GitHub repo page driven by `CITATION.cff`.

Zenodo's GitHub integration auto-mints a DOI per release once the webhook
is enabled. The webhook fires on every GitHub release published *after*
the integration is toggled on (it does **not** retroactively archive older
tags). The metadata Zenodo uses for each deposit is read from
`.zenodo.json` at the repo root; the schema is documented at
<https://developers.zenodo.org/#deposit-metadata>.

## Files already in place (this commit)

| Path | Purpose |
| --- | --- |
| `.zenodo.json` | Zenodo deposit metadata — title, description, creators, keywords, license, related identifiers. |
| `CITATION.cff` | GitHub "Cite this repository" button + machine-readable citation (Citation File Format v1.2.0). |
| `paper/zenodo_setup.md` | This walkthrough. |

## One-time setup (~5 minutes)

1. Visit <https://zenodo.org/account/settings/github/>.
2. Log in via GitHub OAuth — the existing `mahimn01` GitHub account
   already has the permissions Zenodo needs.
3. The GitHub settings page lists every public repo you own. Find
   `mahimn01/reflexive-options` and toggle the integration **ON**.
4. Zenodo registers a webhook at
   `https://zenodo.org/api/hooks/receivers/github/events/`. From this
   point onward, every new GitHub release publishes a draft deposit to
   Zenodo.

## Publishing the first release-with-DOI

1. Push the next tag (e.g. `v0.3.4`) and create a matching GitHub release
   (Releases tab → Draft a new release → choose tag → Publish).
2. Within ~30 seconds, Zenodo's webhook fires. Visit
   <https://zenodo.org/me/uploads> — the new deposit appears as a
   **draft**.
3. Open the draft. Verify that `.zenodo.json` was parsed correctly:
   - Title, description, creators, keywords, license, communities all
     populated.
   - Upload type: Software.
   - Files: the tarball of the release commit, plus `paper/main.pdf`
     (Zenodo packages the entire repo snapshot at the tag).
4. **Before publishing**: replace the ORCID placeholder
   `0000-0000-0000-0000` in `.zenodo.json` with the real iD once
   registered (see `paper/orcid_setup.md`). Do this **before** the
   `v0.3.4` tag is cut so the deposit metadata is right on the first
   pass; otherwise edit the draft creator record directly in the Zenodo
   UI before clicking Publish.
5. Click **Publish**. Zenodo mints a DOI of the form
   `10.5281/zenodo.XXXXXXX` (the numeric portion is assigned at publish
   time). The DOI page becomes the permanent citation target.

## Post-publish: add the badge to README

After the DOI is minted, append a badge near the top of the repo README:

```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
```

Zenodo also exposes a *concept DOI* (resolves to the latest version) and
a *version DOI* (resolves to a specific release). For the README badge,
prefer the **concept DOI** so the link auto-updates across releases. For
the in-paper self-citation (see below), prefer the **version DOI** of the
release that corresponds to the camera-ready code state.

## Verifying the GitHub "Cite this repository" button

1. Wait ~5 minutes after the next push for GitHub's CFF parser to pick
   up the file.
2. Visit <https://github.com/mahimn01/reflexive-options>. The sidebar
   under About should show a **Cite this repository** button.
3. Clicking it pops a modal with APA + BibTeX renderings auto-generated
   from `CITATION.cff`.
4. If the button does not appear, validate the file locally:

   ```bash
   pip install cffconvert
   cffconvert --validate -i CITATION.cff
   ```

## Post-DOI follow-ups (one-shot, after the first published DOI)

These are the cleanup tasks Mahimn should do **once** after the first
Zenodo DOI is minted. Each is intentionally scoped to a single file edit.

### 1. Cite the Zenodo DOI in the manuscript

In `paper/main.tex`, the current self-reference reads:

```
Code at \url{https://github.com/mahimn01/reflexive-options}; pre-registration
anchored to commit \texttt{268c061}.
```

After the DOI is minted, update to (e.g.):

```
Code at \url{https://github.com/mahimn01/reflexive-options}
(Zenodo DOI: \href{https://doi.org/10.5281/zenodo.XXXXXXX}{10.5281/zenodo.XXXXXXX});
pre-registration anchored to commit \texttt{268c061}.
```

Use the **version DOI** of the release whose code state matches the
manuscript's reported numbers — not the concept DOI — so reviewers
reproducing the paper land on the exact frozen snapshot.

### 2. Add the self-citation to references.bib (optional)

If the paper is to carry a formal bibliographic self-reference (some
journals require this for code archives), add an entry to
`paper/references.bib`:

```bibtex
@software{patel2026reflexive_code,
  author    = {Patel, Mahimn},
  title     = {{Reflexivity in Options Markets} (software, v0.3.4)},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v0.3.4},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

Then `\citep{patel2026reflexive_code}` at the relevant point in the
manuscript.

### 3. Update `.zenodo.json` and `CITATION.cff` with the arXiv ID

Once the arXiv submission completes and an arXiv identifier is assigned
(format `YYMM.NNNNN`):

- `.zenodo.json` → replace the `arxiv:XXXX.XXXXX` placeholder in
  `related_identifiers[0].identifier` with the real arXiv ID. The
  `relation` should stay `isSupplementTo` (software supplements the
  preprint) per the Zenodo–arXiv interoperability convention.
- `CITATION.cff` → no change required; the preferred-citation `url` and
  `doi` fields already point at the GitHub repo and (post-mint) the
  Zenodo deposit. Optionally add an `identifiers:` block under
  `preferred-citation`:

  ```yaml
  preferred-citation:
    identifiers:
      - type: other
        value: "arXiv:YYMM.NNNNN"
        description: "arXiv preprint"
  ```

### 4. Tag every paper-relevant release going forward

The Zenodo webhook fires on **every** GitHub release. Reserve
release-with-DOI for substantive versions (theory changes, manuscript
revisions, calibration milestones). For doc-only or test-only changes,
push a tag but **do not** publish a GitHub release — the tag alone does
not trigger Zenodo.

## Reference

- Zenodo GitHub integration docs: <https://help.zenodo.org/docs/profile/linking-github>
- Zenodo deposit metadata schema: <https://developers.zenodo.org/#deposit-metadata>
- Citation File Format v1.2.0 spec: <https://citation-file-format.github.io/>
- GitHub's CFF support: <https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files>
