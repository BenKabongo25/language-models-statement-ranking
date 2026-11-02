import csv,json
from pathlib import Path
from baselines.data import RankingData
from baselines.models.factorization import CD,PITF
from baselines.models.bper import BPER
from baselines.models.cler import CLER


def make(root):
 f=root/'final';f.mkdir(parents=True);fields=['interaction_id','line_index','user_id','item_id','rating','timestamp','split','user_sequence_index','user_sequence_length','item_sequence_index','item_sequence_length','global_sequence_index','statement_ids','aligned_statement_ids','original_statement_ids'];rows=[];iid=0
 for u in ['u1','u2','u3']:
  for t,s in enumerate(['10','11','12','10','11']):
   rows.append({fields[0]:iid,fields[1]:iid,fields[2]:u,fields[3]:'i1',fields[4]:5,fields[5]:t,fields[6]:('test' if t==4 else 'train'),fields[7]:t,fields[8]:5,fields[9]:iid,fields[10]:15,fields[11]:iid,fields[12]:s,fields[13]:s,fields[14]:s});iid+=1
 with (f/'interactions_match.csv').open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
 (f/'id2statement_match.json').write_text(json.dumps({'10':'a','11':'b','12':'c'}))
 with (f/'item_candidates_match.csv').open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=['item_id','statement_ids']);w.writeheader();w.writerow({'item_id':'i1','statement_ids':'10:11:12'})

def cfg(name):return {'model':{'name':name,'dimension':8,'mu':.7,'dropout':.1,'temperature':.2},'training':{'lr':.01,'weight_decay':0.,'epochs':1,'batch_size':4,'num_negatives':1},'sampling':{'candidate_source':'fit_only'}}

def test_dynamic(tmp_path):
 make(tmp_path);d=RankingData(tmp_path);fit=d.temporal_split().train
 for cls,name in [(CD,'cd'),(PITF,'pitf'),(BPER,'bper'),(CLER,'cler')]:
  m=cls(d,cfg(name),40,'cpu');m.fit(fit);assert len(m.score(0,0,d.item_candidates[0]))==3
