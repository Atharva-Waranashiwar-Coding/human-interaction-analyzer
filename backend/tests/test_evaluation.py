from app.schemas import EvaluationRequest
from app.services.evaluation import evaluate


def request(**kw):return EvaluationRequest(covered_behaviors=['carrying'],end_time=10,**kw)
def row(**kw):return dict(id='a',behavior='carrying',start_time=0,end_time=2,participants=['p'],objects=[dict(id='o',class_name='box')],review_status='reviewed',**kw)


def test_no_truth_no_reviewed_no_fabricated_metrics():
    assert not evaluate([],request())['available']
    assert not evaluate([],request(complete_ground_truth=True))['available']


def test_reviewed_predictions_match_independent_truth():
    original=row();pred=row(original_suggestion=original);gt=dict(row(evaluation_role='ground_truth'),id='g')
    result=evaluate([pred,gt],request(complete_ground_truth=True))
    assert result['summary']==dict(true_positives=1,false_positives=0,false_negatives=0,precision=1,recall=1,f1=1)
    pred['review_status']='needs_review'
    result=evaluate([pred,gt],request(complete_ground_truth=True))
    assert result['summary']['false_negatives']==1 and result['summary']['precision'] is None


def test_correction_preserves_original_label_and_false_positive():
    pred=dict(row(original_suggestion=row()),behavior='walking',review_status='rejected')
    result=evaluate([pred],request(complete_ground_truth=True))
    assert result['summary']['false_positives']==1 and result['summary']['recall'] is None


def test_entities_temporal_matching_and_deleted_feedback():
    gt=row(evaluation_role='ground_truth');original=dict(row(),participants=['other'])
    pred=row(original_suggestion=original)
    r=evaluate([pred,gt],request(complete_ground_truth=True))
    assert r['summary']['false_positives']==r['summary']['false_negatives']==1
    log=[dict(annotation_id='deleted',original=original,review_status='rejected')]
    r=evaluate([gt],request(complete_ground_truth=True),log)
    assert r['summary']['false_positives']==1


def test_offline_benchmark_aggregates_scopes_without_inventing_scores():
    from evaluate_fixture import summarize
    original=row()
    truth=dict(row(evaluation_role='ground_truth'),id='truth')
    fixture=dict(request=dict(covered_behaviors=['carrying'],end_time=10,
                              complete_ground_truth=True,scenario='controlled'),
                 annotations=[row(original_suggestion=original),truth])
    report=summarize([fixture,fixture])
    assert report['summary']['true_positives']==2
    assert report['summary']['false_positives']==0
    assert report['summary']['false_negatives']==0
    assert report['summary']['f1']==1
    assert len(report['scopes'])==2


def test_offline_benchmark_requires_explicit_independent_truth_scope():
    from evaluate_fixture import summarize
    import pytest
    with pytest.raises(ValueError,match='no valid evaluation'):
        summarize([dict(request=dict(covered_behaviors=['carrying'],end_time=10),
                        annotations=[row()])])
    with pytest.raises(ValueError,match='nonempty scopes'):
        summarize([])


def test_evaluation_surfaces_exact_unmatched_error_examples():
    original=dict(row(),participants=['wrong'])
    pred=row(original_suggestion=original)
    truth=dict(row(evaluation_role='ground_truth'),id='independent-truth')
    result=evaluate([pred,truth],request(complete_ground_truth=True))
    assert result['summary']['false_positives']==1
    assert result['summary']['false_negatives']==1
    assert result['false_positive_examples'][0]['behavior']=='carrying'
    assert result['missed_ground_truth_examples'][0]['id']=='independent-truth'
    assert result['false_positive_examples'][0]['start_time']==0
