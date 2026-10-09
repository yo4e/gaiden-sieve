"""公式RSSまたは私有cacheから本文を再取得し、hash一致の例だけをローカル復元する。"""
from __future__ import annotations
import argparse,sys,shutil,json,subprocess
from pathlib import Path
from dataclasses import replace
from urllib.parse import urlsplit
from support import HERE,REPO,read_json,digest,verify_protected_files,verify_rows,write_json

def hydrate(ai_gaiden_root,output,feed_cache=None,download=False):
    verify_protected_files();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    ai_root=Path(ai_gaiden_root).resolve()
    commit=subprocess.check_output(['git','-C',str(ai_root),'rev-parse','HEAD'],text=True).strip()
    if commit!='977c9a6b4be35e57f8d945eae7a84c65e5afa03f':raise ValueError('AI-gaiden ingestion commit differs from frozen experiment')
    sys.path.insert(0,str(ai_root))
    from scripts.utils import load_feed_configs
    from scripts.feed_reader import parse_feed_bytes,FeedReader,FeedResult
    from gaiden_sieve.ai_gaiden_shadow import build_shadow_rows
    configs={c.id:replace(c,max_items_per_run=200) for c in load_feed_configs(ai_root/'config/feeds.yml')}
    manifests={name:read_json(HERE/'lineage'/f'{name}.json') for name in ['iteration1','iteration2','quarantine','llm_audit']}
    cache={}
    if feed_cache:
        for p in Path(feed_cache).rglob('*.xml'):cache[digest(p.read_bytes())]=p.read_bytes()
    reader=FeedReader(output/'private-feed-state.json');requested={};texts={};failures=[]
    for manifest in manifests.values():
        for row in manifest['rows']:
            meta=row['record_metadata'];key=(meta['source'],meta['feed_url'])
            if key in requested:continue
            expected='sha256:'+meta['raw_feed_sha256'];payload=cache.get(expected)
            # 現在のRSSへ置換しても、各例のtitle/summary hashが一致しない限り復元しない。
            if payload is None and download:
                url=meta['feed_url'];host=urlsplit(url).hostname
                if urlsplit(url).scheme!='https' or host not in {'blog.google','github.blog','www.microsoft.com','aws.amazon.com'}:
                    raise ValueError('unapproved RSS host in manifest')
                try:
                    response=reader.session.get(url,timeout=(10,30));response.raise_for_status();payload=response.content
                    config=configs[meta['source']];parse_feed_bytes(payload,config)
                    # 生RSSは指定した私有出力内にだけ保存する。
                    raw=output/'private-feed-cache';raw.mkdir(exist_ok=True);(raw/(digest(payload).split(':')[1]+'.xml')).write_bytes(payload)
                except Exception as exc:failures.append(dict(source=meta['source'],feed_url=url,error=str(exc)));payload=None
            requested[key]=payload is not None
            if payload is None:continue
            config=configs[meta['source']];items=parse_feed_bytes(payload,config);result=FeedResult(config,True,False,tuple(items))
            for entry in build_shadow_rows([result],[result],{}):texts.setdefault((entry['source'],entry['source_url']),[]).append(entry)
    status={}
    for name,manifest in manifests.items():
        restored=[];missing=[]
        for row in manifest['rows']:
            meta=row['record_metadata'];candidates=texts.get((meta['source'],meta['source_url']),[])
            matches=[r for r in candidates if digest(r['title'])==row['title_sha256'] and digest(r['summary'][:row['summary_chars']])==row['summary_sha256']]
            if not matches:missing.append(meta['id']);continue
            restored.append(dict(meta,title=matches[0]['title'],summary=matches[0]['summary'][:row['summary_chars']]))
        if missing:
            status[name]=dict(complete=False,missing_ids=missing);continue
        folder=output/name;folder.mkdir(exist_ok=True)
        if name in ('iteration1','iteration2'):
            for base in (REPO/'profiles/ai-gaiden/bootstrap').glob('*.jsonl'):shutil.copy2(base,folder/base.name)
        path=folder/manifest['filename'];path.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in restored),encoding='utf-8')
        verify_rows(path,name);status[name]=dict(complete=True,path=str(path),row_count=len(restored))
    write_json(output/'hydration-status.json',dict(datasets=status,fetch_failures=failures))
    return status

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ai-gaiden-root',type=Path,required=True);parser.add_argument('--output',type=Path,default=REPO/'.diagnosis/issue13-hydrated');parser.add_argument('--feed-cache',type=Path);parser.add_argument('--download-feeds',action='store_true');args=parser.parse_args()
    if not args.feed_cache and not args.download_feeds:parser.error('use --feed-cache or explicitly --download-feeds')
    status=hydrate(args.ai_gaiden_root,args.output,args.feed_cache,args.download_feeds)
    if not all(r['complete'] for r in status.values()):raise SystemExit('Some historical text is no longer obtainable; see hydration-status.json. No substitute labels/text were used.')
    print('Historical labeled datasets hydrated and hashes verified; shadow snapshots require their separate original inputs.')
if __name__=='__main__':main()
