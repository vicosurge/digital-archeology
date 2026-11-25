# Technology fingerprints

The JSON files in this folder are the technology fingerprint database of the
[Wappalyzer](https://github.com/wappalyzer/wappalyzer) project (categories,
groups and technologies), used unmodified apart from the file names. They are
licensed under the **GNU General Public License v3.0**, not under this
repository's MIT license.

Wappalyzer's own repository is no longer public. The same data, kept up to
date under GPL-3.0, is maintained by the community at
[enthec/webappanalyzer](https://github.com/enthec/webappanalyzer).

Only `backend/app/services/techstack.py` reads these files. The rest of the
application does not depend on them, so they can be removed or replaced with
another fingerprint set without affecting anything but tech-stack detection.
