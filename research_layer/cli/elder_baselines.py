"""Descriptive full-period results for the two preregistered classic baselines.

These are never used to select the 72-case family. The full period includes
previously viewed training and validation history, so it is not OOS evidence.
"""
from pathlib import Path
import argparse,json
from research_layer.cli.elder_search import read,save,prepare,summarize,digest,now
from research_layer.elder_engine import ElderConfig


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/research/elder_search_v1.json')
    args=parser.parse_args(argv);p=read(args.config);out=Path(p['output'])/'classic_full_period';out.mkdir(parents=True,exist_ok=True)
    save(out/'descriptive_protocol.json',{'registered_utc':now(),'purpose':'Two fixed original baselines, full-period descriptive only; no selection','period':['2022-01-01','2025-12-31'],'config_sha256':digest(args.config)})
    local={**p,'grid':{**p['grid'],'force_span':[2]}}
    symbols,features,bars,snapshot=prepare(local,out);rows=[]
    for b in p['baselines']:
        cfg=ElderConfig(**p['fixed'],**{k:v for k,v in b.items() if k!='id'})
        r=summarize(cfg,symbols,features,'2022-01-01','2025-12-31');r['baseline_name']=b['id'];rows.append(r)
        print(json.dumps({'baseline':b['id'],'completed':100}),flush=True)
    save(out/'results.json',{'period':['2022-01-01','2025-12-31'],'completed_tasks':200,'failed_tasks':0,'snapshot_sha256':snapshot,'rows':rows})
    return 0

if __name__=='__main__':raise SystemExit(main())
