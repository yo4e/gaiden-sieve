"""学習せず、評価候補の重複群と時刻の境界だけを検査する。"""
from __future__ import annotations
import re
from datetime import datetime
from support import canonical_url,digest,overlaps

NEAR_THRESHOLD=0.60

def normalized_tokens(row):
    # RSSの定型末尾を外す。数字・版番号は共通化し、patch違いの漏洩を保守的に防ぐ。
    summary=re.split(r'\bThe post\b',row.get('summary',''),maxsplit=1)[0]
    text=re.sub(r'\d+(?:\.\d+)*','number',row['title']+' '+summary).casefold()
    return re.findall(r'\w+',text)

def shingles(row):
    words=normalized_tokens(row)
    return {tuple(words[i:i+3]) for i in range(len(words)-2)}

def related(left,right):
    if overlaps(left,right):return 'identity',1.0
    a,b=normalized_tokens(left),normalized_tokens(right)
    if a and a==b:return 'normalized_content',1.0
    sa,sb=shingles(left),shingles(right)
    similarity=len(sa&sb)/len(sa|sb) if sa or sb else 0
    return ('near_template' if min(len(sa),len(sb))>=5 and similarity>=NEAR_THRESHOLD else None),similarity

def groups(rows):
    # 推移的に似た記事も同じ群に置き、source違いの再配信を分割しない。
    parents=list(range(len(rows)));edges=[]
    def root(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    for i in range(len(rows)):
        for j in range(i):
            kind,similarity=related(rows[i],rows[j])
            if kind:parents[root(i)]=root(j);edges.append(dict(left=rows[j]['id'],right=rows[i]['id'],reason=kind,similarity=similarity))
    members={}
    for i,row in enumerate(rows):members.setdefault(root(i),[]).append(row['id'])
    ids={k:digest('\n'.join(sorted(v))) for k,v in members.items()}
    return {row['id']:ids[root(i)] for i,row in enumerate(rows)},edges

def temporal_role(row,cutoff,embargo_days=7):
    # publication timeだけで分ける。取得時刻を記事の新しさと混同しない。
    time=datetime.fromisoformat(row['published_at'].replace('Z','+00:00'))
    boundary=datetime.fromisoformat(cutoff.replace('Z','+00:00'))
    seconds=(time-boundary).total_seconds()
    return 'train_before_cutoff' if seconds<0 else 'embargo' if seconds<embargo_days*86400 else 'future_evaluation'

def group_temporal_roles(rows,group_ids,cutoff,embargo_days=7):
    roles={}
    for row in rows:roles.setdefault(group_ids[row['id']],set()).add(temporal_role(row,cutoff,embargo_days))
    # 同じ群がcutoff/embargoを跨いだら全群を外し、似た記事を時間分割の両側へ置かない。
    decisions={group:next(iter(values)) if len(values)==1 and 'embargo' not in values else 'purged' for group,values in roles.items()}
    return {row['id']:decisions[group_ids[row['id']]] for row in rows}
