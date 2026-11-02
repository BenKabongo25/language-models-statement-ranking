from baselines.metrics import metrics_at_k

def test_metrics_binary():
    m=metrics_at_k([1,2,3,4],{1,3},2)
    assert m['precision@2']==0.5
    assert m['recall@2']==0.5
    assert m['hr@2']==1.0
    assert 0 < m['ndcg@2'] <= 1
