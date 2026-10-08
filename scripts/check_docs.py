"""Validate local reader links and draft bundle assets without network access."""
import argparse
import json
from pathlib import Path
import re
from urllib.parse import unquote,urlsplit


def check(root,blog=False):
    paths=sorted(root.glob('*/index.md')) if blog else [p for p in [root/'README.md',root/'CHANGELOG.md',root/'results/README.md'] if p.exists()]+sorted((root/'docs').rglob('*.md'))
    failures=[]
    for path in paths:
        raw=path.read_text()
        if blog:
            if 'draft: true' not in raw:failures.append(f'{path}: expected draft flag')
            if '\u2014' in raw:failures.append(f'{path}: em dash')
            if len(re.findall(r'^```',raw,re.M))%2:failures.append(f'{path}: unbalanced code fences')
        text=re.sub(r'```.*?```','',raw,flags=re.S)
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
            relref=re.fullmatch(r'\{\{<\s*relref\s+["\']([^"\']+)["\']\s*>\}\}',target.strip())
            target=relref.group(1) if relref else target.split(' "')[0].strip('<>')
            parts=urlsplit(target)
            if parts.scheme or parts.netloc or target.startswith('/'):
                continue
            dest=(path.parent/unquote(parts.path)).resolve() if parts.path else path.resolve()
            if dest.is_dir():
                if blog:dest=dest/'index.md'
                else:continue
            if not dest.exists():
                failures.append(f'{path}: missing {target}')
            elif parts.fragment and dest.suffix=='.md':
                contents=re.sub(r'```.*?```','',dest.read_text(),flags=re.S)
                anchors=set(re.findall(r'(?:id|name)=["\']([^"\']+)',contents))
                counts={}
                for heading in re.findall(r'^#{1,6} (.+)$',contents,re.M):
                    slug=re.sub(r'[^\w\- ]','',heading.lower()).replace(' ','-')
                    count=counts.get(slug,0);counts[slug]=count+1
                    anchors.add(slug+(f'-{count}' if count else ''))
                if unquote(parts.fragment) not in anchors:
                    failures.append(f'{path}: missing heading anchor {target}')
    return {'scope':'Local inline Markdown links, relative Hugo relref targets and heading anchors; external URLs, generated Hugo rendering and reference-style links are not checked.',
            'files':len(paths),'failures':failures,'passed':not failures}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--blog',action='store_true');a=p.parse_args()
    result=check(a.root,a.blog);print(json.dumps(result,indent=2));raise SystemExit(0 if result['passed'] else 1)

if __name__=='__main__':main()
