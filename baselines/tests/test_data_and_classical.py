import csv, json
from pathlib import Path
from baselines.data import RankingData
from baselines.models.classical import GlobalPop, UserPop, ItemPop


def _make(root: Path):
    f=root/'final';f.mkdir(parents=True)
    fields=['interaction_id','line_index','user_id','item_id','rating','timestamp','split','user_sequence_index','user_sequence_length','item_sequence_index','item_sequence_length','global_sequence_index','statement_ids','aligned_statement_ids','original_statement_ids']
    rows=[];iid=0
    for u in ['u1','u2']:
      for t,(i,s) in enumerate([('i1','10'),('i1','11'),('i1','10:12'),('i1','11'),('i1','12')]):
        rows.append(dict(interaction_id=iid,line_index=iid,user_id=u,item_id=i,rating=5,timestamp=t,split=('test' if t==4 else 'train'),user_sequence_index=t,user_sequence_length=5,item_sequence_index=iid,item_sequence_length=10,global_sequence_index=iid,statement_ids=s,aligned_statement_ids=s,original_statement_ids=s));iid+=1
    with (f/'interactions_match.csv').open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
    (f/'id2statement_match.json').write_text(json.dumps({'10':'a','11':'b','12':'c'}))
    with (f/'item_candidates_match.csv').open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=['item_id','statement_ids']);w.writeheader();w.writerow({'item_id':'i1','statement_ids':'10:11:12'})


def test_loader_and_pop(tmp_path):
    _make(tmp_path);d=RankingData(tmp_path);sp=d.temporal_split();assert len(sp.test)==2 and len(sp.val)==2
    m=GlobalPop(d,{'model':{}},0);m.fit(sp.train+sp.val);c=d.item_candidates[0];assert len(m.score(0,0,c))==3

def test_explicit_upstream_test_is_authoritative(tmp_path):
    _make(tmp_path)
    d = RankingData(tmp_path)
    sp = d.temporal_split()
    # Each user has 5 events: [0,1,2] train, 3 validation, 4 upstream test.
    assert len(sp.train) == 6
    assert len(sp.val) == 2
    assert len(sp.test) == 2
    assert all(x.timestamp == 4 for x in sp.test)
    assert all(x.timestamp == 3 for x in sp.val)
