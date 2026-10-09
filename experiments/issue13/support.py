"""公開manifestで私有入力の整合性を確認する。本文の公開は行わない。"""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
from urllib.parse import urlsplit,urlunsplit,parse_qsl,urlencode

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]

def digest(value):
    if isinstance(value,str):value=value.encode('utf-8')
    return 'sha256:'+hashlib.sha256(value).hexdigest()

def write_json(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')

def verify_protected_files():
    mismatches=[name for name,expected in read_json(HERE/'protected-file-hashes.json').items() if digest((REPO/name).read_bytes())!=expected]
    if mismatches:raise ValueError('frozen core/profile/evaluation files changed: '+', '.join(mismatches))

def verify_rows(path,manifest_name):
    manifest=read_json(HERE/'lineage'/f'{manifest_name}.json');path=Path(path)
    if digest(path.read_bytes())!=manifest['file_sha256']:raise ValueError(f'{manifest_name}: file hash mismatch; historical replay unavailable')
    rows=read_jsonl(path)
    if len(rows)!=len(manifest['rows']):raise ValueError(f'{manifest_name}: row count mismatch')
    for row,description in zip(rows,manifest['rows']):
        if digest(row['title'])!=description['title_sha256'] or digest(row.get('summary',''))!=description['summary_sha256']:
            raise ValueError(f'{manifest_name}: input text hash mismatch')
        if {k:v for k,v in row.items() if k not in ('title','summary')}!=description['record_metadata']:
            raise ValueError(f'{manifest_name}: metadata mismatch')
    return rows

def canonical_url(row):
    parts=urlsplit(row.get('source_url',''))
    return urlunsplit((parts.scheme.lower(),parts.netloc.lower(),parts.path.rstrip('/'),urlencode(sorted((k,v) for k,v in parse_qsl(parts.query) if not k.startswith('utm_'))),''))

def identity(row):
    return [row['id'],canonical_url(row),re.sub(r'\s+',' ',row['title']).strip().casefold(),re.sub(r'\s+',' ',row['title']+' '+row.get('summary','')[:500]).strip().casefold()]

def overlaps(left,right):
    return any(a and a==b for a,b in zip(identity(left),identity(right)))

def verify_boundaries(training_path,audit_path=None):
    training=sum((read_jsonl(p) for p in sorted(Path(training_path).glob('*.jsonl'))),[])
    holdout=sum((read_jsonl(p) for p in sorted((REPO/'profiles/ai-gaiden/real-holdout').glob('*.jsonl'))),[])
    # gold.jsonlにURLがないため、gold-reviewの同じ確定記事も保護する。
    gold=read_jsonl(REPO/'profiles/ai-gaiden/gold-review.jsonl')
    audit=read_jsonl(audit_path) if audit_path else []
    for name,rows in [('holdout',holdout),('gold',gold),('llm_audit',audit)]:
        if any(overlaps(a,b) for a in training for b in rows):raise ValueError(f'training overlaps {name}')
    return training,holdout,gold,audit
