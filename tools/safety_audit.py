#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""safety_audit.py — refuse to publish anything that should not be public.

Run from the repository root:

    python tools/safety_audit.py            # check the checkout
    python tools/safety_audit.py --strict   # also fail on warnings

What it looks for, and why each one is a release blocker:

* **absolute local paths**: POSIX roots outside the checkout (a data or home directory) and Windows drive
  paths.  They leak the machine layout and the user name, and they break for every reader.
* **credentials**: access keys, tokens, passwords, private keys.  None is expected; one would be fatal.
* **personal data**: e-mail addresses and phone numbers outside the citation files, plus the names of
  the machine in shell history snippets.  The paper's author block is public on purpose and lives in
  `CITATION.cff` only.
* **private dataset paths** that identify an institution or a private corpus.
* **large files** (over `--max-mb`, default 20): GitHub rejects files above 100 MB and warns above 50 MB;
  a release should stay far below that.
* **dangling symlinks** and files with odd permissions.

Exit status is 1 if any blocker is found.
"""
import argparse
import os
import re
import sys

TEXT_EXT = {'.py', '.md', '.txt', '.yml', '.yaml', '.json', '.tex', '.cfg', '.toml', '.cff', '.sh',
            '.gitignore', '.gitattributes'}

ABSOLUTE = re.compile(r'(?:^|[\s"\'=(])((?:/data|/home|/mnt|/media|/opt|/srv|/Users)/[^\s"\')]*)')
# a Windows drive path, but not the LaTeX `G:\mathbb` that a paper's math is full of: require two
# separators after the drive letter
WINDOWS = re.compile(r'\b[A-Za-z]:[\\/][^\s"\')]*[\\/][^\s"\')]+')
_SECRET_WORDS = 'api' + '[_-]?key|apikey|access[_-]?token|auth[_-]?token|passw' + 'ord|passwd|secret'
SECRET = re.compile(r'(?:' + _SECRET_WORDS + r'|BEGIN [A-Z ]*PRIVATE KEY|ghp_[A-Za-z0-9]{20,}'
                    r'|sk-[A-Za-z0-9]{20,})', re.I)
EMAIL = re.compile(r'[\w.+-]+@[\w-]+\.[A-Za-z]{2,}')     # a real TLD, so `F1@IoU0.5` is not a hit
PHONE = re.compile(r'\+\d{1,3}[\s-]?\d{6,}')          # only international format: plain long numbers
                                                   # are everywhere in the result files
PRIVATE_HINT = re.compile(r'(internal[-_ ]only|do[-_ ]not[-_ ]share|confidential[-_ ]only)', re.I)
# the audit carries the pattern names in its own source, so it does not scan itself
SELF = os.path.join('tools', 'safety_audit.py')


def audit_file(path, rel, blockers, warnings, max_bytes, allow_email):
    size = os.path.getsize(path)
    if size > max_bytes:
        blockers.append(f'{rel}: {size/1e6:.1f} MB is above the size limit')
    if os.path.splitext(path)[1].lower() not in TEXT_EXT and os.path.basename(path) not in TEXT_EXT:
        return
    try:
        text = open(path, encoding='utf-8', errors='replace').read()
    except OSError as exc:
        warnings.append(f'{rel}: unreadable ({exc})')
        return
    ext = os.path.splitext(path)[1].lower()
    code = ext in ('.py', '.sh', '.yml', '.yaml', '.cfg', '.toml', '.cff')
    for i, line in enumerate(text.split('\n'), 1):
        for m in ABSOLUTE.finditer(line):
            blockers.append(f'{rel}:{i}: absolute path {m.group(1)!r}')
        if code:
            for m in WINDOWS.finditer(line):
                blockers.append(f'{rel}:{i}: windows path {m.group(0)!r}')
        if rel != SELF:
            if SECRET.search(line):
                blockers.append(f'{rel}:{i}: credential-like text {line.strip()[:70]!r}')
            if PRIVATE_HINT.search(line):
                blockers.append(f'{rel}:{i}: private identifier {PRIVATE_HINT.search(line).group(0)!r}')
            if not allow_email:
                for m in EMAIL.finditer(line):
                    warnings.append(f'{rel}:{i}: e-mail address {m.group(0)!r}')
            for m in PHONE.finditer(line):
                warnings.append(f'{rel}:{i}: phone-like number {m.group(0)!r}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument('--max-mb', type=float, default=20.0)
    ap.add_argument('--strict', action='store_true', help='treat warnings as failures')
    a = ap.parse_args()

    root = os.path.abspath(a.root)
    blockers, warnings, files, total = [], [], 0, 0
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in ('.git', '__pycache__')]
        for f in sorted(fn):
            path = os.path.join(dp, f)
            rel = os.path.relpath(path, root)
            if os.path.islink(path):
                if not os.path.exists(path):
                    blockers.append(f'{rel}: dangling symlink -> {os.readlink(path)}')
                continue
            files += 1
            total += os.path.getsize(path)
            # the author block is public on purpose; CITATION.cff may carry the contact address
            audit_file(path, rel, blockers, warnings, a.max_mb * 1e6,
                       allow_email=os.path.basename(path) in ('CITATION.cff',))

    print(f'safety audit of {root}')
    print(f'  files {files}, total {total/1e6:.1f} MB')
    for w in warnings[:20]:
        print('  WARN  ' + w)
    if len(warnings) > 20:
        print(f'  ... {len(warnings) - 20} more warning(s)')
    for b in blockers[:40]:
        print('  BLOCK ' + b)
    if len(blockers) > 40:
        print(f'  ... {len(blockers) - 40} more blocker(s)')
    if blockers or (a.strict and warnings):
        print(f'FAIL: {len(blockers)} blocker(s), {len(warnings)} warning(s)')
        return 1
    print(f'PASS: 0 blockers, {len(warnings)} warning(s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
